#include "vm.h"
#include <iostream>
#include <cstdlib>
#include <cstring>

#ifdef ENABLE_HTTP
#include "http_server.cpp"
#endif

int main(int argc, char** argv) {
#ifdef ENABLE_HTTP
    if (argc >= 2 && strcmp(argv[1], "--server") == 0) {
        int port = (argc > 2) ? std::atoi(argv[2]) : 8080;
        HTTPServer server(port);
        server.start();
        std::cout << "vicVM server running on port " << port << ". Press Enter to stop.\n";
        std::cin.get();
        server.stop();
        return 0;
    }
#endif

    // Normal VM mode
    if (argc < 2) {
        std::cerr << "Usage: vicVM <guest-image> [ram-size-MB] [cores] [backing-file]\n";
        return 1;
    }
    const char* image = argv[1];
    uint64_t ram_mb = (argc > 2) ? std::atoll(argv[2]) : 64;
    uint32_t cores  = (argc > 3) ? std::atoi(argv[3]) : 4;
    const char* backing = (argc > 4) ? argv[4] : nullptr;

    VM vm;
    if (!vm.init(ram_mb * 1024 * 1024, image, cores, backing)) {
        std::cerr << "VM init failed.\n";
        return 1;
    }
    vm.run();
    return 0;
}
