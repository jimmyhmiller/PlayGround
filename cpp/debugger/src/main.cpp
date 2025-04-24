#include <iostream>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach-o/loader.h>
#include <mach-o/nlist.h>
#include <mach-o/dyld_images.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <spawn.h>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <cstring>
#include <cerrno>
#include <iomanip>
#include <optional>



////////////////////////
// ADDED HEADERS
////////////////////////
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <unistd.h>
#include <mach/exception_types.h>
#include <shared_mutex>

extern char **environ;



//////////////////////////////////////////////////////////
// Data structures
//////////////////////////////////////////////////////////
struct LoadedImage {
    std::string path;
    uintptr_t baseAddress;
};

struct Breakpoint {
    mach_vm_address_t addr;
    std::vector<uint8_t> originalBytes;
    int is_debugger_info;
};


template<typename T>
class ThreadSafeContainer {
public:
    void add_element(const T& value) {
        std::unique_lock lock(mutex_);
        data_.push_back(value);
    }

    void print_elements() const {
        std::shared_lock lock(mutex_);
        for (const auto& elem : data_) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    template<typename Func>
    std::optional<T> find_element(Func func) const {
        std::shared_lock lock(mutex_);
        for (const auto& elem : data_) {
            if (func(elem)) {
                return elem;  // Return a copy of the element
            }
        }
        return std::nullopt;
    }

private:
    mutable std::shared_mutex mutex_;
    std::vector<T> data_;
};

ThreadSafeContainer<Breakpoint> breakpoints;


//////////////////////////////////////////////////////////
// Launch a process in a paused state, with arguments
//////////////////////////////////////////////////////////

pid_t launch_process_paused(const char* program, const std::vector<const char*>& args) {
    pid_t pid;
    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_START_SUSPENDED);

    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(program));
    for (auto arg : args) {
        argv.push_back(const_cast<char*>(arg));
    }
    argv.push_back(nullptr);

    if (posix_spawn(&pid, program, nullptr, &attr, argv.data(), environ) != 0) {
        std::cerr << "Error: Unable to launch process " << program << std::endl;
        exit(1);
    }
    return pid;
}

//////////////////////////////////////////////////////////
// Attach / detach / resume / suspend
//////////////////////////////////////////////////////////

task_t get_task_for_pid(pid_t pid) {
    task_t task;
    if (task_for_pid(mach_task_self(), pid, &task) != KERN_SUCCESS) {
        std::cerr << "Error: Unable to get task for pid " << pid << std::endl;
        exit(1);
    }
    return task;
}

void resume_task(task_t task) {
    kern_return_t kr = task_resume(task);
    if (kr != KERN_SUCCESS) {
        std::cerr << "Error: Unable to resume task " << kr << std::endl;
        exit(1);
    }
}

void suspend_task(task_t task) {
    kern_return_t kr = task_suspend(task);
    if (kr != KERN_SUCCESS) {
        std::cerr << "Error: Unable to suspend task " << kr << std::endl;
        exit(1);
    }
}

//////////////////////////////////////////////////////////
// Reading / writing remote memory
//////////////////////////////////////////////////////////

void read_memory(task_t task, mach_vm_address_t address, void* buffer, size_t size) {
    vm_size_t out_size;
    kern_return_t kr = vm_read_overwrite(task, address, size, (mach_vm_address_t)buffer, &out_size);
    if (kr != KERN_SUCCESS || out_size != size) {
        std::cerr << "Error: Unable to read memory at 0x" << std::hex << address
                  << ", kr=" << kr << std::endl;
        exit(1);
    }
}

// We'll need to temporarily change protections to be able to write.
void write_memory(task_t task, mach_vm_address_t address, const void* buffer, size_t size) {
    // 1) Align to the page
    mach_vm_address_t page_start = address & ~(mach_vm_address_t)(vm_page_size - 1);
    mach_vm_size_t page_offset = address - page_start;
    mach_vm_size_t total_size  = page_offset + size;
    
    // get current protection flags using vm_region
    vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object_name;
    mach_vm_region(task, &page_start, &total_size, VM_REGION_BASIC_INFO_64,
                   (vm_region_info_t)&info, &count, &object_name);


    

    // 2) Add VM_PROT_WRITE | VM_PROT_EXECUTE
    kern_return_t kr = mach_vm_protect(task, page_start, total_size, false,
                                       VM_PROT_READ | VM_PROT_WRITE | VM_PROT_COPY);
    if (kr != KERN_SUCCESS) {
        std::cerr << "Error: Unable to mach_vm_protect at 0x" << std::hex << page_start
                  << ", kr=" << kr << std::endl;
        exit(1);
    }

    // 3) Write
    kr = vm_write(task, address, (vm_address_t)buffer, (mach_msg_type_number_t)size);
    if (kr != KERN_SUCCESS) {
        std::cerr << "Error: Unable to vm_write at 0x" << std::hex << address
                  << ", kr=" << kr << std::endl;
        exit(1);
    }
    
    // 4) Restore protections
    kr = mach_vm_protect(task, page_start, total_size, false,
                         info.protection);
    if (kr != KERN_SUCCESS) {
        std::cerr << "Error: Unable to mach_vm_protect at 0x" << std::hex << page_start
                  << ", kr=" << kr << std::endl;
        exit(1);
    }
        
}

std::string read_remote_string(task_t task, mach_vm_address_t addr, size_t max_len=1024) {
    std::vector<char> buf(max_len, '\0');
    read_memory(task, addr, buf.data(), max_len - 1);
    buf[max_len - 1] = '\0';
    return std::string(buf.data());
}

//////////////////////////////////////////////////////////
// Access dyld_all_image_infos to get loaded libraries
//////////////////////////////////////////////////////////

mach_vm_address_t get_dyld_info_address(task_t task) {
    struct task_dyld_info dyld_info;
    mach_msg_type_number_t count = TASK_DYLD_INFO_COUNT;
    kern_return_t kr = task_info(task, TASK_DYLD_INFO, (task_info_t)&dyld_info, &count);
    if (kr != KERN_SUCCESS) {
        std::cerr << "Error: Unable to get dyld info, kr=" << kr << std::endl;
        exit(1);
    }
    return dyld_info.all_image_info_addr;
}

std::vector<LoadedImage> parse_remote_images(task_t task, mach_vm_address_t dyld_info_addr) {
    std::vector<LoadedImage> result;
    dyld_all_image_infos aii;
    read_memory(task, dyld_info_addr, &aii, sizeof(aii));

    uint32_t count = aii.infoArrayCount;
    mach_vm_address_t array_addr = (mach_vm_address_t)aii.infoArray;

    for (uint32_t i = 0; i < count; i++) {
        dyld_image_info remote_info;
        mach_vm_address_t cur = array_addr + i * sizeof(dyld_image_info);
        read_memory(task, cur, &remote_info, sizeof(remote_info));

        std::string path = "[unknown]";
        if (remote_info.imageFilePath) {
            path = read_remote_string(task, (mach_vm_address_t)remote_info.imageFilePath);
        }
        mach_vm_address_t load_addr = (mach_vm_address_t)remote_info.imageLoadAddress;

        LoadedImage img;
        img.path = path;
        img.baseAddress = (uintptr_t)load_addr;
        result.push_back(img);
    }
    return result;
}

//////////////////////////////////////////////////////////
// Parse Mach-O for a symbol
//////////////////////////////////////////////////////////

mach_vm_address_t find_symbol_in_remote_image(task_t task,
                                              mach_vm_address_t remote_base,
                                              const std::string& symbolName)
{
    // Read Mach-O header
    mach_header_64 mh;
    read_memory(task, remote_base, &mh, sizeof(mh));
    if (mh.magic != MH_MAGIC_64 && mh.magic != MH_CIGAM_64) {
        return 0; // not a 64-bit Mach-O
    }

    uint32_t ncmds = mh.ncmds;
    mach_vm_address_t cursor = remote_base + sizeof(mh);

    mach_vm_address_t linkedit_vmaddr = 0;
    uint64_t linkedit_fileoff = 0;
    uint64_t slide = 0;
    uint32_t symoff = 0, nsyms = 0, stroff = 0;

    uint64_t text_vmaddr = 0;

    // Parse commands
    for (uint32_t i = 0; i < ncmds; i++) {
        load_command lc;
        read_memory(task, cursor, &lc, sizeof(lc));

        if (lc.cmd == LC_SEGMENT_64) {
            segment_command_64 seg;
            read_memory(task, cursor, &seg, sizeof(seg));
            if (strcmp(seg.segname, "__TEXT") == 0) {
                text_vmaddr = seg.vmaddr;
            }
            if (strcmp(seg.segname, "__LINKEDIT") == 0) {
                linkedit_vmaddr = seg.vmaddr;
                linkedit_fileoff = seg.fileoff;
            }
        }
        else if (lc.cmd == LC_SYMTAB) {
            symtab_command sc;
            read_memory(task, cursor, &sc, sizeof(sc));
            symoff = sc.symoff;
            nsyms = sc.nsyms;
            stroff = sc.stroff;
        }

        cursor += lc.cmdsize;
    }

    if (!text_vmaddr || !linkedit_vmaddr || !symoff || !nsyms || !stroff) {
        return 0; // can't parse
    }

    // Slide calculation
    slide = remote_base - text_vmaddr;

    mach_vm_address_t linkedit_start = remote_base + (linkedit_vmaddr - text_vmaddr);
    mach_vm_address_t symtab_addr = linkedit_start + (symoff - linkedit_fileoff);
    mach_vm_address_t strtab_addr = linkedit_start + (stroff - linkedit_fileoff);

    // Read nlist_64 array
    size_t size_nlist = sizeof(nlist_64) * nsyms;
    std::vector<nlist_64> nl(nsyms);
    read_memory(task, symtab_addr, nl.data(), size_nlist);

    for (uint32_t i = 0; i < nsyms; i++) {
        uint32_t stridx = nl[i].n_un.n_strx;
        if (stridx == 0) continue;

        char nameBuf[1024];
        memset(nameBuf, 0, sizeof(nameBuf));
        read_memory(task, strtab_addr + stridx, nameBuf, sizeof(nameBuf) - 1);

        const char* rawName = (nameBuf[0] == '_') ? &nameBuf[1] : nameBuf;
        if (symbolName == rawName) {
            return nl[i].n_value + slide;
        }
    }
    return 0;
}

//////////////////////////////////////////////////////////
// Insert a software breakpoint
//////////////////////////////////////////////////////////

Breakpoint set_breakpoint(task_t task, mach_vm_address_t address, int is_debugger_info=0) {
    Breakpoint bp;
    bp.addr = address;
    bp.is_debugger_info = is_debugger_info;

    // On ARM64, we patch BRK #0 => 0xd4200000
    bp.originalBytes.resize(4);
    read_memory(task, address, bp.originalBytes.data(), 4);

    uint32_t brkInstr = 0xd4200000;
    write_memory(task, address, &brkInstr, sizeof(brkInstr));

    return bp;
}

void clear_breakpoint(task_t task, const Breakpoint& bp) {
    // Restore original bytes
    write_memory(task, bp.addr, bp.originalBytes.data(), bp.originalBytes.size());
}


void handle_trap(mach_port_t task) {
    mach_msg_type_number_t count = TASK_BASIC_INFO_COUNT;
    task_basic_info info;
    kern_return_t kr = task_info(task, TASK_BASIC_INFO, (task_info_t)&info, &count);

    if (kr == KERN_SUCCESS) {
        std::cout << "Trap handled: Suspended task " << task << "\n";
        task_suspend(task);
    } else {
        std::cerr << "Error getting task info: " << kr << "\n";
    }
}

void change_pc_of_child(task_t task) {
    arm_thread_state64_t state;
    mach_msg_type_number_t count = ARM_THREAD_STATE64_COUNT;

    // Get the thread state
    thread_act_t thread;
    thread_array_t thread_list;
    mach_msg_type_number_t thread_count;

    if (task_threads(task, &thread_list, &thread_count) != KERN_SUCCESS) {
        std::cerr << "Error getting threads from task" << std::endl;
        exit(1);
    }

    thread = thread_list[0];  // Get the first thread

    if (thread_get_state(thread, ARM_THREAD_STATE64, (thread_state_t)&state, &count) != KERN_SUCCESS) {
        std::cerr << "Error getting thread state" << std::endl;
        exit(1);
    }

    std::cout << "Original PC: 0x" << std::hex << state.__pc << std::endl;

    // Change the Program Counter (PC)
    state.__pc += 4;  // Skip one instruction (4 bytes on ARM)

    if (thread_set_state(thread, ARM_THREAD_STATE64, (thread_state_t)&state, count) != KERN_SUCCESS) {
        std::cerr << "Error setting thread state" << std::endl;
        exit(1);
    }

    std::cout << "Updated PC: 0x" << std::hex << state.__pc << std::endl;
}

void exception_handler_thread(mach_port_t exceptionPort) {
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        union {
            mach_msg_header_t hdr;
            char buf[1024];
        } message;
        kern_return_t kr = mach_msg(&message.hdr, MACH_RCV_MSG, 0, sizeof(message),
                                      exceptionPort, MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
        if (kr != KERN_SUCCESS) {
            std::cerr << "mach_msg (receive) error: " << mach_error_string(kr) << "\n";
            continue;
        }
        std::cerr << "Exception received. Handling...\n";
        struct reply_message {
            mach_msg_header_t Head;
            NDR_record_t NDR;
            kern_return_t RetCode;
        } reply;
        std::memset(&reply, 0, sizeof(reply));
        reply.Head.msgh_bits = MACH_MSGH_BITS(MACH_MSGH_BITS_REMOTE(message.hdr.msgh_bits), 0);
        reply.Head.msgh_size = sizeof(reply);
        reply.Head.msgh_remote_port = message.hdr.msgh_remote_port;
        reply.Head.msgh_local_port = MACH_PORT_NULL;
        reply.Head.msgh_id = message.hdr.msgh_id + 100;
        reply.NDR = NDR_record;
        reply.RetCode = KERN_SUCCESS;
        kr = mach_msg(&reply.Head, MACH_SEND_MSG, reply.Head.msgh_size, 0,
                      MACH_PORT_NULL, MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
        if (kr != KERN_SUCCESS) {
            std::cerr << "mach_msg (send) error: " << mach_error_string(kr) << "\n";
        }
    }
}



void set_hardware_breakpoint(thread_t thread, mach_vm_address_t address) {
    arm_debug_state64_t dbg_state;
    mach_msg_type_number_t count = ARM_DEBUG_STATE64_COUNT;
    kern_return_t kr = thread_get_state(thread, ARM_DEBUG_STATE64,
                                        reinterpret_cast<thread_state_t>(&dbg_state), &count);
    if (kr != KERN_SUCCESS) {
        std::cerr << "Error getting debug state: " << mach_error_string(kr) << "\n";
        return;
    }

    // Use hardware breakpoint 0 (DBGBVR0 and DBGBCR0).
    dbg_state.__bvr[0] = address;        // Set the breakpoint address.
    dbg_state.__bcr[0] = (1 << 0) |      // Enable the breakpoint.
                         (0b1111 << 5);  // Match any instruction type (load/store/execution).
    
    kr = thread_set_state(thread, ARM_DEBUG_STATE64,
                          reinterpret_cast<thread_state_t>(&dbg_state), count);
    if (kr != KERN_SUCCESS) {
        std::cerr << "Error setting debug state: " << mach_error_string(kr) << "\n";
    } else {
        // std::cerr << "Hardware breakpoint set at 0x" << std::hex << address << "\n";
    }
}

void clear_hardware_breakpoint(thread_t thread) {
    arm_debug_state64_t dbg_state;
    mach_msg_type_number_t count = ARM_DEBUG_STATE64_COUNT;
    kern_return_t kr = thread_get_state(thread, ARM_DEBUG_STATE64,
                                        reinterpret_cast<thread_state_t>(&dbg_state), &count);
    if (kr != KERN_SUCCESS) {
        std::cerr << "Error getting debug state: " << mach_error_string(kr) << "\n";
        return;
    }

    // Clear hardware breakpoint 0.
    dbg_state.__bcr[0] = 0;  // Disable the breakpoint.

    kr = thread_set_state(thread, ARM_DEBUG_STATE64,
                          reinterpret_cast<thread_state_t>(&dbg_state), count);
    if (kr != KERN_SUCCESS) {
        std::cerr << "Error clearing debug state: " << mach_error_string(kr) << "\n";
    } else {
        // std::cerr << "Hardware breakpoint cleared.\n";
    }
}

// --- Assume these types and functions are defined elsewhere ---
// struct Image { mach_vm_address_t baseAddress; std::string path; };
// struct Breakpoint { /* platform-specific fields */ };
//
// pid_t launch_process_paused(const char*, const std::vector<const char*>&);
// task_t get_task_for_pid(pid_t);
// void suspend_task(task_t);
// void resume_task(task_t);
// mach_vm_address_t get_dyld_info_address(task_t);
// std::vector<Image> parse_remote_images(task_t, mach_vm_address_t);
// mach_vm_address_t find_symbol_in_remote_image(task_t, mach_vm_address_t, const std::string&);
// void change_pc_of_child(task_t);
// Breakpoint set_breakpoint(task_t, mach_vm_address_t);
// void clear_breakpoint(task_t, Breakpoint);

// For ARM, define our own exception message structure since mach_exception_raise_request_t is unavailable.
typedef struct {
    mach_msg_header_t Head;
    mach_msg_body_t msgh_body;
    mach_msg_port_descriptor_t thread;
    mach_msg_port_descriptor_t task;
    NDR_record_t NDR;
    exception_type_t exception;
    mach_msg_type_number_t codeCnt;
    mach_exception_data_type_t code[2];
} my_mach_exception_raise_request_t;



// Encoding an unsigned integer v (of any type excepting u8/i8) works as follows:

// If u < 251, encode it as a single byte with that value.
// If 251 <= u < 2**16, encode it as a literal byte 251, followed by a u16 with value u.
// If 2**16 <= u < 2**32, encode it as a literal byte 252, followed by a u32 with value u.
// If 2**32 <= u < 2**64, encode it as a literal byte 253, followed by a u64 with value u.
// If 2**64 <= u < 2**128, encode it as a literal byte 254, followed by a u128 with value u.
// usize is being encoded/decoded as a u64 and isize is being encoded/decoded as a i64.

// See the documentation of VarintEncoding for more information.


struct BincodeParser {
    uint8_t* data;
    size_t size;
    size_t offset;

    BincodeParser(uint8_t* data, size_t size) : data(data), size(size), offset(0) {}
    

    int readInt() {
        int value = data[offset];
        if (value < 251) {
            offset++;
            return value;
        } else {
            std::cerr << "Error: First byte is >= 251, not printing\n";
            std::cout << "First byte: " << std::hex << value << "\n";
            return -1;  // Indicate that parsing failed
        }
        return value;
    }

    void skip_one() {
        if (offset + 1 > size) {
            std::cerr << "Error: Not enough data to skip int\n";
            return;
        }
        offset += 1;
    }

    uint64_t readUInt() {
        int first_byte = static_cast<int>(data[offset]);
        if (first_byte < 251) {
            // read_int
            int value = this->readInt();
            return value;
        }
        if (first_byte == 251) {
            offset++;
            // Read the next 2 bytes as a short
            int value = 0;
            for (int i = 0; i < 2; ++i) {
                value |= (static_cast<int>(data[offset + i]) << (i * 8));
            }
            offset += 2;
            return value;
        }
        if (first_byte == 252) {
            offset++;
            // Read the next 4 bytes as an int
            int value = 0;
            for (int i = 0; i < 4; ++i) {
                value |= (static_cast<int>(data[offset + i]) << (i * 8));
            }
            offset += 4;
            return value;
        }
        if (first_byte == 253) {
            offset++;
            // Read the next 8 bytes as an unsigned long long
            unsigned long long value = 0;
            for (int i = 0; i < 8; ++i) {
                value |= (static_cast<unsigned long long>(data[offset + i]) << (i * 8));
            }
            offset += 8;
            return static_cast<long>(value);
        } else {
            std::cout << "First byte is not 253, not printing.\n";
            std::cout << "First byte: " << std::hex << first_byte << "\n";
            return -1;  // Indicate that parsing failed
        }
    }

    std::string readString() {
        int length = readInt();
        if (length == -1 || offset + length > size) {
            std::cerr << "Error: Not enough data to read string\n";
            return "";
        }
        std::string str(reinterpret_cast<char*>(data + offset), length);
        offset += length;
        return str;
    }
    std::string read_rest_as_string() {
        if (offset >= size) {
            std::cerr << "Error: No more data to read\n";
            return "";
        }
        std::string str(reinterpret_cast<char*>(data + offset), size - offset);
        offset = size;  // Move offset to the end
        return str;
    }

    void print_each_byte() {
        for (size_t i = 0; i < size; ++i) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(data[i]) << " ";
        }
        std::cout << "\n";
    }

};

void read_memory_from_task(task_t task, uint64_t address, uint64_t length) {
    mach_vm_size_t data_size = length;
    mach_vm_address_t target_address = address;
    vm_offset_t data;
    mach_msg_type_number_t data_count;

    kern_return_t kr = mach_vm_read(task, target_address, data_size, &data, &data_count);
    if (kr != KERN_SUCCESS) {
        std::cerr << "Error reading memory at 0x" << std::hex << address
                  << ": " << mach_error_string(kr) << "\n";
        return;
    }

    if (data_count == 0) {
        std::cerr << "No data read from target address.\n";
        return;
    }


    BincodeParser parser(reinterpret_cast<uint8_t*>(data), data_count);
    
    std::string kind = parser.readString();
    // We already know the enum discriminate
    parser.skip_one();
    if (kind == "user_function") {
        std::string function_name = parser.readString();
        std::cout << "Function name: " << function_name << "\n";
        if (function_name == "beagle.core/swap!") {
            uint64_t pointer = parser.readUInt();
            uint64_t length = parser.readUInt();
            uint64_t num_arguments = parser.readUInt();
            auto bp = set_breakpoint(task, pointer);
            breakpoints.add_element(bp);
            std::cout << "Breakpoint set at 0x" << std::hex << pointer << "\n";
            std::cout << "Pointer: " << std::hex << pointer << "\n";
            std::cout << "Length: " << std::hex << length << "\n";
            std::cout << "Num arguments: " << std::hex << num_arguments << "\n";
        }
    }
    

    // Deallocate the data read from the target task's memory space
    kr = vm_deallocate(mach_task_self(), data, data_count);
    if (kr != KERN_SUCCESS) {
        std::cerr << "Error deallocating memory: " << mach_error_string(kr) << "\n";
    }
}

void read_x0_x1_memory(thread_t thread, task_t task) {
    arm_thread_state64_t state;
    mach_msg_type_number_t count = ARM_THREAD_STATE64_COUNT;
    kern_return_t kr = thread_get_state(thread, ARM_THREAD_STATE64,
                                        reinterpret_cast<thread_state_t>(&state), &count);
    if (kr != KERN_SUCCESS) {
        std::cerr << "Error getting thread state: " << mach_error_string(kr) << "\n";
        return;
    }

    uint64_t x0 = state.__x[0];  // Pointer to memory
    uint64_t x1 = state.__x[1];  // Length

    // std::cout << "x0 (pointer): 0x" << std::hex << x0 << "\n";
    // std::cout << "x1 (length): " << std::dec << x1 << " bytes\n";

    if (x1 > 0) {
        read_memory_from_task(task, x0, x1);
    } else {
        std::cerr << "Length (x1) is zero, nothing to read.\n";
    }
}

kern_return_t enable_single_step(thread_t thread) {
    arm_debug_state64_t debug_state;
    mach_msg_type_number_t count = ARM_DEBUG_STATE64_COUNT;

    kern_return_t kr = thread_get_state(thread, ARM_DEBUG_STATE64,
                                        (thread_state_t)&debug_state, &count);
    if (kr != KERN_SUCCESS) return kr;

    // Set MDSCR_EL1.SS (bit 0) to enable single-step
    debug_state.__mdscr_el1 |= (1 << 0);

    kr = thread_set_state(thread, ARM_DEBUG_STATE64,
                          (thread_state_t)&debug_state, ARM_DEBUG_STATE64_COUNT);
    return kr;
}

kern_return_t disable_single_step(thread_t thread) {
    arm_debug_state64_t debug_state;
    mach_msg_type_number_t count = ARM_DEBUG_STATE64_COUNT;

    kern_return_t kr = thread_get_state(thread, ARM_DEBUG_STATE64,
                                        (thread_state_t)&debug_state, &count);
    if (kr != KERN_SUCCESS) return kr;

    // Clear MDSCR_EL1.SS
    debug_state.__mdscr_el1 &= ~(1 << 0);

    kr = thread_set_state(thread, ARM_DEBUG_STATE64,
                          (thread_state_t)&debug_state, ARM_DEBUG_STATE64_COUNT);
    return kr;
}

// For ARM, assume exception code 1 for breakpoint and 2 for single-step.
 
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <program> <symbol> [args...]\n";
        return 1;
    }
    
    const char* program = argv[1];
    std::string symbol = argv[2];
    std::vector<const char*> extraArgs;
    for (int i = 3; i < argc; i++) {
        extraArgs.push_back(argv[i]);
    }
    
    pid_t pid = launch_process_paused(program, extraArgs);
    std::cout << "Launched '" << program << "' with PID=" << pid << " (suspended)\n";
    
    if (ptrace(PT_ATTACHEXC, pid, 0, 0) < 0) {
        perror("ptrace(PT_ATTACHEXC)");
        return 1;
    }
    
    task_t task = get_task_for_pid(pid);
    
    // Set up Mach exception handling.
    mach_port_t exceptionPort;
    kern_return_t kr = mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &exceptionPort);
    if (kr != KERN_SUCCESS) {
        std::cerr << "mach_port_allocate error: " << mach_error_string(kr) << "\n";
        return 1;
    }
    kr = mach_port_insert_right(mach_task_self(), exceptionPort, exceptionPort, MACH_MSG_TYPE_MAKE_SEND);
    if (kr != KERN_SUCCESS) {
        std::cerr << "mach_port_insert_right error: " << mach_error_string(kr) << "\n";
        return 1;
    }
    kr = task_set_exception_ports(task, EXC_MASK_BREAKPOINT ,
                                  exceptionPort, EXCEPTION_DEFAULT, THREAD_STATE_NONE);
    if (kr != KERN_SUCCESS) {
        std::cerr << "task_set_exception_ports error: " << mach_error_string(kr) << "\n";
        return 1;
    }
    kr = task_set_exception_ports(task, EXC_MASK_BAD_ACCESS ,
                                  exceptionPort, EXCEPTION_DEFAULT, THREAD_STATE_NONE);
    if (kr != KERN_SUCCESS) {
        std::cerr << "task_set_exception_ports error: " << mach_error_string(kr) << "\n";
        return 1;
    }

    
    suspend_task(task);
    resume_task(task);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    suspend_task(task);
    
    mach_vm_address_t dyld_info_addr = get_dyld_info_address(task);
    auto images = parse_remote_images(task, dyld_info_addr);
    
    std::cout << "Looking for symbol '" << symbol << "'...\n";
    mach_vm_address_t foundAddr = 0;
    for (auto& img : images) {
        mach_vm_address_t sym = find_symbol_in_remote_image(task, img.baseAddress, symbol);
        if (sym) {
            foundAddr = sym;
            std::cout << "Found '" << symbol << "' in: " << img.path
                      << " at 0x" << std::hex << sym << "\n";
            break;
        }
    }
    
    if (!foundAddr) {
        std::cerr << "Symbol '" << symbol << "' not found.\n";
        return 0;
    }
    
    // Insert the breakpoint.
    auto bp = set_breakpoint(task, foundAddr, 1);
    breakpoints.add_element(bp);
    std::cout << "Breakpoint set at 0x" << std::hex << foundAddr << "\n";
    
    // Exception handler thread:
    // On breakpoint exception (code 1): clear the breakpoint and issue a single step.
    // On single-step exception (code 2): reinsert the breakpoint and continue.

    std::thread([&task, exceptionPort, pid]() {
        bool is_stepping = false;
        std::chrono::high_resolution_clock::time_point start_time, end_time;
    
        while (true) {
            union {
                mach_msg_header_t hdr;
                char buf[1024];
            } message;
            start_time = std::chrono::high_resolution_clock::now();
            kern_return_t kr = mach_msg(&message.hdr, MACH_RCV_MSG, 0, sizeof(message),
                                        exceptionPort, MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
            if (kr != KERN_SUCCESS) {
                std::cerr << "mach_msg (receive) error: " << mach_error_string(kr) << "\n";
                continue;
            }

            // std::cerr << "Exception received. Handling...\n";
    
            // Correctly interpret the exception message structure
            auto* req = reinterpret_cast<my_mach_exception_raise_request_t*>(&message);
            thread_t thread = req->thread.name;
            arm_thread_state64_t state;
            mach_msg_type_number_t count = ARM_THREAD_STATE64_COUNT;
            kr = thread_get_state(thread, ARM_THREAD_STATE64,
                                  reinterpret_cast<thread_state_t>(&state), &count);
            if (kr != KERN_SUCCESS) {
                std::cerr << "Error getting thread state: " << mach_error_string(kr) << "\n";
                continue;
            }


            auto address = state.__pc;

            // std::cerr << "Exception at address: 0x" << std::hex << address << "\n";
    
            if (req->exception == EXC_BREAKPOINT || req->exception == EXC_BAD_ACCESS) {
                if (!is_stepping) {
    
                    // std::cerr << "Breakpoint hit at address: 0x" << std::hex << address << "\n";

                    // Get the current PC to calculate the next instruction
                    arm_thread_state64_t state;
                    mach_msg_type_number_t count = ARM_THREAD_STATE64_COUNT;
                    kr = thread_get_state(thread, ARM_THREAD_STATE64,
                                          reinterpret_cast<thread_state_t>(&state), &count);
                    if (kr != KERN_SUCCESS) {
                        std::cerr << "Error getting thread state: " << mach_error_string(kr) << "\n";
                        continue;
                    }

                    enable_single_step(thread);
                    // std::cerr << "Single-step enabled.\n";

                    auto address = state.__pc;

                    // pass a lambda to find_elements
                    auto bp = breakpoints.find_element([&](const Breakpoint& b) {
                        return b.addr == address;
                    });

                    if (!bp) {
                        std::cerr << "No breakpoint found at 0x" << std::hex << address << "\n";
                        continue;
                    }

                    // Clear the software breakpoint
                    clear_breakpoint(task, *bp);
                    // std::cerr << "Software breakpoint cleared at 0x" << std::hex << address << "\n";
                    if (bp->is_debugger_info) {
                        read_x0_x1_memory(thread, task);
                    }

                    // Calculate the next instruction address
                    // mach_vm_address_t next_addr = state.__pc + 4;  // ARM64: assume 4-byte instruction
    
                    // // Set hardware breakpoint at the next instruction
                    // set_hardware_breakpoint(thread, next_addr);
                    is_stepping = true;
    
                } else {
                    // This is the hardware breakpoint hit
                    // std::cerr << "Hardware single-step breakpoint hit! Restoring software breakpoint.\n";

                    // Clear the hardware breakpoint
                    // clear_hardware_breakpoint(thread);
                    
                    // std::cerr << "Hit single step breakpoint!\n";

                    // Reinsert the original software breakpoint
                    is_stepping = false;

                    arm_thread_state64_t state;
                    mach_msg_type_number_t count = ARM_THREAD_STATE64_COUNT;
                    kr = thread_get_state(thread, ARM_THREAD_STATE64,
                                          reinterpret_cast<thread_state_t>(&state), &count);
                    if (kr != KERN_SUCCESS) {
                        std::cerr << "Error getting thread state: " << mach_error_string(kr) << "\n";
                        continue;
                    }

                    disable_single_step(thread);
                    // std::cerr << "Single-step disabled.\n";
    
                    auto address = state.__pc - 4;;

                    // pass a lambda to find_elements
                    auto bp = breakpoints.find_element([&](const Breakpoint& b) {
                        return b.addr == address;
                    });


                    if (!bp) {
                        std::cerr << "No breakpoint found at 0x" << std::hex << address << "\n";
                        continue;
                    }
                    // Reinsert the software breakpoint
                    set_breakpoint(task, (*bp).addr);
                    // std::cerr << "Software breakpoint restored at 0x" << std::hex << address << "\n";
                }
            } else {
                std::cerr << "Unknown exception received: " << req->exception << "\n";
            }
    
            // Reply to the exception message
            struct reply_message {
                mach_msg_header_t Head;
                NDR_record_t NDR;
                kern_return_t RetCode;
            } reply;
            std::memset(&reply, 0, sizeof(reply));
            reply.Head.msgh_bits = MACH_MSGH_BITS(MACH_MSGH_BITS_REMOTE(message.hdr.msgh_bits), 0);
            reply.Head.msgh_size = sizeof(reply);
            reply.Head.msgh_remote_port = message.hdr.msgh_remote_port;
            reply.Head.msgh_local_port = MACH_PORT_NULL;
            reply.Head.msgh_id = message.hdr.msgh_id + 100;
            reply.NDR = NDR_record;
            reply.RetCode = KERN_SUCCESS;
    
            kr = mach_msg(&reply.Head, MACH_SEND_MSG, reply.Head.msgh_size, 0,
                          MACH_PORT_NULL, MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
            if (kr != KERN_SUCCESS) {
                std::cerr << "mach_msg (send) error: " << mach_error_string(kr) << "\n";
            }

            if (is_stepping) {
                end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_seconds = end_time - start_time;
                std::cerr << "Elapsed time: " << elapsed_seconds.count() << "s\n";
            }
        }
    }).detach();
    
    resume_task(task);
    std::cout << "Process resumed. Waiting for child process to exit...\n";
    
    int status;
    while (true) {
        pid_t result = waitpid(pid, &status, 0);
        if (result == -1) {
            if (errno == EINTR) {
                continue;
            } else if (errno == ECHILD) {
                std::cerr << "No more child processes to wait for.\n";
                break;
            } else {
                std::cerr << "Error waiting for child process: " << strerror(errno) << "\n";
                break;
            }
        }
        if (WIFEXITED(status)) {
            std::cout << "Child exited with status: " << WEXITSTATUS(status) << "\n";
            break;
        }
        if (WIFSIGNALED(status)) {
            std::cerr << "Child process terminated by signal: " << std::to_string(WTERMSIG(status)) << "\n";
            break;
        }
    }
    
    return 0;
}
