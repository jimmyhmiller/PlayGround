
#include <__expected/expected.h>
#include <__expected/unexpected.h>
#include <iostream>
#include <mach/arm/kern_return.h>
#include <mach/arm/vm_types.h>
#include <mach/kern_return.h>
#include <spawn.h>
#include <expected>
#include <mach/mach.h>
#include <mach-o/dyld_images.h>
#include <mach/mach_vm.h>
#include <vector>
#include <sys/ptrace.h>
#include "generated/mach_exc_server.h"
#include "generated/mach_excServer.c"
#include <dlfcn.h>
#include <mach-o/loader.h>



template<typename... Args>
void println(Args&&... args) {
    ((std::cout << std::forward<Args>(args) << ' '), ...) << '\n';
}

void *(*m_dyld_process_info_create)(task_t task,
  uint64_t timestamp,
  kern_return_t *kernelError);

void (*m_dyld_process_info_for_each_image)(void *info,
         void (^callback)(uint64_t machHeaderAddress,
                          const uuid_t uuid,
                          const char *path));

void (*m_dyld_process_info_release)(void *info);





struct Task {
  task_read_t task;
  mach_port_t exception_port;
  pid_t pid;
  mach_port_t last_thread_port;

  static auto from_pid(pid_t pid) -> std::expected<Task, kern_return_t> {
    task_t task_id;
    auto kr = task_for_pid(mach_task_self(), pid, &task_id);
    if (kr != KERN_SUCCESS) {
      return std::unexpected(kr);
    }
    return Task {
      .task = task_id,
      .pid = pid
    };
  }

  auto resume() -> std::expected<void, kern_return_t> {
    auto kr = task_resume(task);
    if (kr != KERN_SUCCESS) {
      return std::unexpected(kr);
    }
    return std::expected<void, kern_return_t>{};
  }

  auto suspend() -> std::expected<void, kern_return_t> {
    auto kr = task_suspend(task);
    if (kr != KERN_SUCCESS) {
      return std::unexpected(kr);
    }
    return std::expected<void, kern_return_t>{};
  }

  auto setup_mach_task() -> std::expected<void, kern_return_t> {
    mach_port_t exception_port;
    kern_return_t kr = mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &exception_port);
    if (kr != KERN_SUCCESS) {
        return std::unexpected(kr);
    }
    kr = mach_port_insert_right(mach_task_self(), exception_port, exception_port, MACH_MSG_TYPE_MAKE_SEND);
    if (kr != KERN_SUCCESS) {
        return std::unexpected(kr);
    }
    kr = task_set_exception_ports(task,
      EXC_MASK_ALL,
      exception_port,
      EXCEPTION_STATE_IDENTITY | MACH_EXCEPTION_CODES,
      ARM_THREAD_STATE64);
    if (kr != KERN_SUCCESS) {
        return std::unexpected(kr);
    }
    this->exception_port = exception_port;
    return std::expected<void, kern_return_t>{};
  };

  auto get_dylib_info() -> std::expected<dyld_all_image_infos, kern_return_t>  {
    struct task_dyld_info dyld_info;
    mach_msg_type_number_t count = TASK_DYLD_INFO_COUNT;
    kern_return_t kr = task_info(task, TASK_DYLD_INFO, (task_info_t)&dyld_info, &count);
    if (kr != KERN_SUCCESS) {
      return std::unexpected(kr);
    }
    dyld_all_image_infos aii;
    this->read_memory(dyld_info.all_image_info_addr, &aii, sizeof(aii)).value();
    return aii;
  }

  auto read_memory(mach_vm_address_t address, void* buffer, size_t size) -> std::expected<void, kern_return_t> {
    vm_size_t out_size;
    kern_return_t kr = vm_read_overwrite(this->task, address, size, (mach_vm_address_t)buffer, &out_size);
    if (kr != KERN_SUCCESS || out_size != size) {
      return std::unexpected(kr);
    }
    return std::expected<void, kern_return_t>{};
  }

  auto read_mach_header(mach_vm_address_t remote_base) -> std::expected<mach_header_64, int> {
    mach_header_64 mh;
    this->read_memory(remote_base, &mh, sizeof(mh)).value();
    if (mh.magic != MH_MAGIC_64 && mh.magic != MH_CIGAM_64) {
        return std::unexpected(-1); // not a 64-bit Mach-O
    }
    return mh;
  }
};

// GLOBAL
Task task;

typedef void *dyld_process_info;

void on_process_created() {
  kern_return_t krt;
  dyld_process_info info = m_dyld_process_info_create(task.task, 0, &krt);
  if (krt != KERN_SUCCESS) {
    println("Unable to retrieve dyld_process_info_create information");
  }


  // TODO: I get main here. I can then 
  if (info) {
    m_dyld_process_info_for_each_image(
      info,
      ^(uint64_t mach_header_addr, const uuid_t uuid, const char *path) {
        char *base_name = strrchr((char*)path, '/');
        base_name = (base_name) ? base_name + 1 : (char*)path;
        println(mach_header_addr); 
        println((char*)base_name);
        if (strcmp(base_name, "main")) {
          // this gets us the module base address. We now need to parse the entry point of the module
        }
      });

    m_dyld_process_info_release(info);
  }
}


auto spawn_process_paused(const char* path, const std::vector<const char*>& args) -> pid_t {
  pid_t pid;
  posix_spawnattr_t attr;
  posix_spawnattr_init(&attr);
  posix_spawnattr_setflags(&attr, POSIX_SPAWN_START_SUSPENDED);

  std::vector<char*> argv;
  argv.push_back(const_cast<char*>(path));
    for (auto arg : args) {
        argv.push_back(const_cast<char*>(arg));
    }
    argv.push_back(nullptr);
  posix_spawn(&pid, path, nullptr, &attr, argv.data(), nullptr);
  return pid;
}

auto catch_mach_exception_raise_state_identity
(
	mach_port_t exception_port,
	mach_port_t thread_port,
	mach_port_t task_port,
	exception_type_t exception_type,
	mach_exception_data_t code,
	mach_msg_type_number_t code_cnt,
	int *flavor,
	thread_state_t old_state,
	mach_msg_type_number_t old_state_cnt,
	thread_state_t new_state,
	mach_msg_type_number_t *new_state_cnt
) -> kern_return_t {

  std::memcpy(new_state, old_state, old_state_cnt * sizeof(natural_t));
  *new_state_cnt = old_state_cnt;

  if (exception_type == EXC_SOFTWARE &&
      code_cnt >= 2 &&
      code[0] == EXC_SOFT_SIGNAL &&
      code[1] == SIGSTOP) {
        task.last_thread_port = thread_port;

        // TODO: I need to use the _dyld_process_info_create
        // so I can read where main is before things load
        // and put a breakpoint there
        // The dylibs will be loaded, I can find debugger_info
        // and do what I need to do
        // task.get_dylib_info().value();
        return KERN_SUCCESS;
  }

  println("Unhandled exception");
  return KERN_FAILURE;
}

auto catch_mach_exception_raise
(
	mach_port_t exception_port,
	mach_port_t thread,
	mach_port_t task,
	exception_type_t exception,
	mach_exception_data_t code,
	mach_msg_type_number_t codeCnt
) -> kern_return_t  {
  println("HERE!!");
  return 0;
}

auto catch_mach_exception_raise_state
(
	mach_port_t exception_port,
	exception_type_t exception,
	const mach_exception_data_t code,
	mach_msg_type_number_t codeCnt,
	int *flavor,
	const thread_state_t old_state,
	mach_msg_type_number_t old_stateCnt,
	thread_state_t new_state,
	mach_msg_type_number_t *new_stateCnt
) -> kern_return_t {
  println("OTHER!");
  return 0;
}

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

  m_dyld_process_info_create =
  (void *(*)(task_t task, uint64_t timestamp, kern_return_t * kernelError))
      dlsym(RTLD_DEFAULT, "_dyld_process_info_create");
m_dyld_process_info_for_each_image =
  (void (*)(void *info, void (^)(uint64_t machHeaderAddress,
                                 const uuid_t uuid, const char *path)))
      dlsym(RTLD_DEFAULT, "_dyld_process_info_for_each_image");
m_dyld_process_info_release =
  (void (*)(void *info))dlsym(RTLD_DEFAULT, "_dyld_process_info_release");

  int pid = spawn_process_paused("/Users/jimmyhmiller/Documents/Code/beagle/target/debug/main", extraArgs);
  task = Task::from_pid(pid).value();
  task.setup_mach_task().value();
  auto ptrace_ret = ptrace(PT_ATTACHEXC, task.pid, 0, 0);
  mach_msg_header_t *request_buffer = (mach_msg_header_t *)malloc(sizeof(union __RequestUnion__exc_subsystem));
  mach_msg_header_t *reply_buffer = (mach_msg_header_t *)malloc(sizeof(union __ReplyUnion__exc_subsystem));
  while (true) {
    mach_msg_return_t result = mach_msg(request_buffer,
      MACH_RCV_MSG | MACH_RCV_TIMEOUT | MACH_RCV_INTERRUPT,
      0,
      sizeof(union __RequestUnion__exc_subsystem),
      task.exception_port,
      10000,
      MACH_PORT_NULL);

        if (result == MACH_RCV_TIMED_OUT) {
          continue;
        }

        // Dispatch to your exception handler
        if (!mach_exc_server(request_buffer, reply_buffer)) {
            std::cerr << "mach_exc_server could not handle message" << std::endl;
        }


        mach_msg_return_t send_result = mach_msg(reply_buffer,
          MACH_SEND_MSG,
          reply_buffer->msgh_size,
          0,
          MACH_PORT_NULL,
          MACH_MSG_TIMEOUT_NONE,
          MACH_PORT_NULL);

        if (send_result != MACH_MSG_SUCCESS) {
            std::cerr << "mach_msg send failed: " << mach_error_string(send_result) << std::endl;
        }

        // Is this deterministic? It seems to work, but it feels like there has to be a better way
        // regardless, it works for now. I've been trying and trying to get a determinsitic process
        // but I can't seem to figure out exactly how to tell when main has been loaded
        // other than it seems this.
        task.suspend().value();
        on_process_created();
        task.resume().value();

        if (ptrace(PT_THUPDATE, task.pid, (caddr_t)(uintptr_t)task.last_thread_port, SIGCONT) == -1) {
          perror("ptrace PT_THUPDATE");
        }
  }
  println(pid);
  println("TEST");
  return 0;
}
