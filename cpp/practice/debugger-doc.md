# Debugger Framework Documentation Outline

Below is an outline that documents both the general program flow and summarizes the external API functions (from system and Mach libraries as well as standard C/C++ libraries) used in the code. This documentation isn’t a line‐by‐line rewrite but a conceptual “skeleton” that highlights the major components and responsibilities.

---

## 1. General Flow of the Code

1. **Process Launch and Setup:**  
   - **Spawning a Paused Process:**  
     The program starts by launching a new process in a paused state using the POSIX process creation API. It builds an argument list and uses functions (e.g., *posix_spawn*) with attributes that ensure the process does not begin execution immediately.  
     
   - **Attaching and Obtaining a Task Port:**  
     After launching, the program attaches to the child process using a system call (*ptrace*) and retrieves a Mach “task” port for the target process. This port acts as the gateway for further operations on the target (e.g., reading/writing memory).

2. **Memory and Image Handling:**  
   - **Reading Remote Memory:**  
     The code contains helper functions that use the Mach virtual memory API to read from the process’s memory space.  
   - **Parsing Loaded Images:**  
     It accesses the dyld (dynamic loader) information structure to retrieve a list of loaded images (libraries and executables). Then it iterates over each image to search for a specified symbol within the Mach-O headers.

3. **Breakpoint Insertion and Management:**  
   - **Setting a Breakpoint:**  
     Once the target symbol is located, the program inserts a software breakpoint by writing an architecture-specific breakpoint instruction to the memory location. The original bytes are saved to allow restoration later.
   - **Hardware Breakpoint & Single-Step Handling:**  
     The code also demonstrates how to enable and disable single-stepping via ARM-specific debug registers. This is used together with hardware breakpoint support to manage the flow when a breakpoint is hit.

4. **Exception Handling with Mach Ports:**  
   - **Establishing Exception Ports:**  
     The process’s exception handling is set up by allocating and inserting rights for a Mach port. The target task is configured to send breakpoint and bad-access exceptions to this port.
   - **Dedicated Exception Handler Thread:**  
     A separate thread enters a loop to receive and respond to exceptions via Mach messaging. When a breakpoint (or single-step) is detected, it uses thread state API calls to adjust the program counter and switch between software and hardware breakpoint modes.
   - **Responding to Exceptions:**  
     Depending on the type of exception (e.g., a software breakpoint hit or a single-step event), the handler will clear or reinsert breakpoints and may read additional target memory (for debugger–related information).

5. **Resuming and Finishing:**  
   - **Task Resumption:**  
     The process is resumed after breakpoints are set. After handling exceptions and (if applicable) stepping through instructions, the program resumes normal execution of the target.  
   - **Process Waiting:**  
     Finally, the parent process waits for the child process to exit, handling various termination cases (normal exit or signal termination).

---

## 2. API Documentation for External Library Functions

Below is a summary of key external functions, grouped by their libraries and purposes.

### A. POSIX Process Creation and Control

- **`posix_spawn` (from `<spawn.h>`):**  
  **Purpose:** Creates a new process to execute the specified program.  
  **Parameters:**  
  - Pointer to a `pid_t` variable to receive the process ID.
  - Path to the executable.
  - Optional file action or attributes structures.
  - An attribute object (initialized with related functions) that can specify flags (e.g., to suspend at startup).
  - Array of argument strings and the environment pointer (e.g., `environ`).  
  **Notes:** This function is used here with the `POSIX_SPAWN_START_SUSPENDED` flag so that the process starts in a suspended state.

- **`posix_spawnattr_init` / `posix_spawnattr_setflags` (from `<spawn.h>`):**  
  **Purpose:**  
  - `posix_spawnattr_init` initializes a spawn attributes object.  
  - `posix_spawnattr_setflags` sets attribute flags (like starting the process in a suspended state).  
  **Usage:** These functions configure the spawn behavior before calling `posix_spawn`.

- **`ptrace` (from `<sys/ptrace.h>`):**  
  **Purpose:** Provides an interface to observe and control the execution of another process.  
  **Usage in Code:** It is invoked with the `PT_ATTACHEXC` operation to attach to the newly spawned process for debugging purposes.

- **`waitpid` (from `<sys/wait.h>`):**  
  **Purpose:** Waits for a state change in a child process (e.g., exit or signal termination).  
  **Parameters:**  
  - The process ID to wait for.
  - A pointer to an integer to store exit status.
  - Flags to control waiting behavior (in this code, called in a blocking manner).

### B. Mach Kernel and Virtual Memory APIs

- **`task_for_pid` (from `<mach/mach.h>`):**  
  **Purpose:** Retrieves the Mach “task” (a handle representing a process) for a given process ID.  
  **Parameters:**  
  - The current task (obtained via `mach_task_self()`).
  - The target process’s PID.
  - Pointer to a task variable to store the result.

- **`mach_task_self` (from `<mach/mach.h>`):**  
  **Purpose:** Returns a send right to the current process’s task port (used to operate on self or as a parameter for other calls).

- **Memory Reading/Writing Functions:**  
  - **`vm_read_overwrite` (from `<mach/mach_vm.h>`):**  
    **Purpose:** Reads memory from a remote task into a buffer provided by the caller.  
    **Parameters:**  
    - Target task, source address in the target, size, destination buffer, and variable to return number of bytes read.
  
  - **`vm_write` (from `<mach/mach_vm.h>`):**  
    **Purpose:** Writes data from the local process to a specified address in the target task’s memory.  
    **Usage:** Used after temporarily changing memory protections.
  
  - **`mach_vm_protect` (from `<mach/mach_vm.h>`):**  
    **Purpose:** Modifies the protection attributes (read, write, execute) of a region in the target task’s address space.  
    **Usage:** Temporarily set to allow write operations (and later restored).
  
  - **`mach_vm_region` (from `<mach/mach_vm.h>`):**  
    **Purpose:** Retrieves information about a region in the task’s memory, such as current protection flags.

  - **`mach_vm_read` (from `<mach/mach_vm.h>`):**  
    **Purpose:** Reads a block of memory from the target task into a newly allocated buffer.  
    **Usage:** This is used later when reading encoded data from the target’s memory.

- **Task Control:**  
  - **`task_suspend` / `task_resume` (from `<mach/mach.h>`):**  
    **Purpose:**  
    - `task_suspend` halts execution of all threads in the target task.
    - `task_resume` restarts the execution of a suspended task.
  
  - **`task_info` (from `<mach/mach.h>`):**  
    **Purpose:** Retrieves various information about a task. In this code, it is used to obtain the dynamic loader info.

- **Exception Handling:**  
  - **`task_set_exception_ports` (from `<mach/mach.h>`):**  
    **Purpose:** Configures a task to send exceptions (like breakpoints or bad memory accesses) to a specified Mach port.  
    **Usage:** The code installs exception handlers for breakpoint and bad-access events.
  
  - **`mach_port_allocate` / `mach_port_insert_right` (from `<mach/mach.h>`):**  
    **Purpose:**  
    - `mach_port_allocate` creates a new Mach port.
    - `mach_port_insert_right` associates send rights with a port so that it can be used for communication.
  
  - **`mach_msg` (from `<mach/mach.h>`):**  
    **Purpose:** Sends and receives messages over Mach ports.  
    **Usage:** Employed in the dedicated exception handling thread to receive exception notifications and send replies.
  
  - **`mach_error_string` (from `<mach/mach_error.h>`):**  
    **Purpose:** Converts Mach kernel error codes into human-readable strings.

### C. Thread State and Hardware Debugging APIs

- **`thread_get_state` / `thread_set_state` (from `<mach/mach.h>`):**  
  **Purpose:**  
  - `thread_get_state` retrieves register values or other state information for a specific thread.
  - `thread_set_state` sets registers or modifies the thread’s state (e.g., to adjust the program counter).
  **Usage:**  
  - These functions are central to implementing both single-stepping (by toggling hardware flags) and updating the program counter after handling a breakpoint.
  
- **`task_threads` (from `<mach/mach.h>`):**  
  **Purpose:** Retrieves a list of threads belonging to a task.  
  **Usage:** The code uses it to select a thread for which the state (registers) can be inspected and modified.

- **ARM Debug and Thread State Structures (from `<mach/arm/thread_state.h>` or similar):**  
  **`arm_thread_state64_t`:** Represents the general-purpose register state for an ARM64 thread.  
  **`arm_debug_state64_t`:** Holds registers related to debugging (hardware breakpoints, single-stepping).  
  **Usage:** These structures are manipulated via the thread state functions to control breakpoint behavior.

### D. C++ Standard Libraries and Concurrency

- **Standard I/O Streams (`std::cout`, `std::cerr`):**  
  **Purpose:** Output logs, debug messages, and error messages.
  
- **`std::thread`, `std::this_thread::sleep_for`, and `<chrono>`:**  
  **Purpose:**  
  - Manage concurrent execution (e.g., the exception handling loop runs on a separate thread).
  - Introduce delays (for sleeping and throttling operations).

- **`std::vector`, `std::string` and Related Utilities:**  
  **Usage:** Used throughout for dynamic storage (e.g., storing breakpoints, buffering memory reads) and string manipulation.

- **`std::shared_mutex` and the `ThreadSafeContainer` Class:**  
  **Purpose:** Provides thread-safe access and modification of shared data (like the breakpoint list) using a combination of shared and unique locks.

---

## Summary

In essence, this code implements a lightweight debugging and instrumentation framework that:
- Launches and suspends a target process,
- Attaches to it to manipulate its memory and registers,
- Parses the loaded Mach-O images to locate symbols,
- Inserts software/hardware breakpoints,
- Processes exceptions via a dedicated handler thread, and
- Coordinates the manipulation of thread states to support operations like single-stepping.

The external functions are drawn mainly from the POSIX, Mach Kernel, and C++ standard libraries. They provide the building blocks for process control, memory manipulation, thread management, and asynchronous messaging required to implement debugger-like functionality.