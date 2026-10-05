#include <iostream>

// Tells the compiler this function is defined elsewhere with C linkage
extern "C" void run_kernel();

int main() {
    std::cout << "[CPU] Starting main program." << std::endl;
    
    // Call the wrapper function in kernel.cu
    run_kernel();
    
    std::cout << "[CPU] Exiting main program." << std::endl;
    return 0;
}
