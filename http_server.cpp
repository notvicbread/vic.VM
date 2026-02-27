#ifdef ENABLE_HTTP

#include "vm.h"
#include <cstring>
#include <string>
#include <map>
#include <thread>
#include <mutex>
#include <sstream>
#include <iostream>

#ifdef _WIN32
    // Windows socket includes already in platform.h
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #define SOCKET int
    #define INVALID_SOCKET -1
    #define closesocket close
#endif

// Very simple HTTP server (one thread, blocking)
class HTTPServer {
public:
    HTTPServer(int port) : port_(port), running_(false) {}
    ~HTTPServer() { stop(); }

    void start() {
        running_ = true;
        server_thread_ = std::thread(&HTTPServer::run, this);
    }

    void stop() {
        running_ = false;
        if (server_thread_.joinable()) server_thread_.join();
    }

private:
    int port_;
    std::thread server_thread_;
    std::atomic<bool> running_;
    std::map<int, std::unique_ptr<VM>> vms_;
    std::mutex vms_mtx_;
    int next_vm_id_ = 1;

    void run() {
        SOCKET listen_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd == INVALID_SOCKET) return;
        int opt = 1;
        setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
        sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port_);
        if (bind(listen_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
            closesocket(listen_fd);
            return;
        }
        listen(listen_fd, 10);

        while (running_) {
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(listen_fd, &fds);
            timeval tv = {1, 0};
            if (select(listen_fd+1, &fds, NULL, NULL, &tv) <= 0) continue;
            sockaddr_in client_addr;
            socklen_t client_len = sizeof(client_addr);
            SOCKET client_fd = accept(listen_fd, (sockaddr*)&client_addr, &client_len);
            if (client_fd == INVALID_SOCKET) continue;

            char buf[4096];
            int n = recv(client_fd, buf, sizeof(buf)-1, 0);
            if (n > 0) {
                buf[n] = 0;
                handle_request(client_fd, buf);
            }
            closesocket(client_fd);
        }
        closesocket(listen_fd);
    }

    void handle_request(SOCKET client_fd, const char* request) {
        std::string req(request);
        std::string response;
        if (req.find("POST /vm") == 0) {
            // Crude JSON parsing (real project would use a library)
            std::string image = "guest.bin";
            uint64_t ram = 64;
            uint32_t cores = 4;
            // ... (skip parsing for brevity)
            {
                std::lock_guard<std::mutex> lock(vms_mtx_);
                int id = next_vm_id_++;
                auto vm = std::make_unique<VM>();
                if (vm->init(ram * 1024 * 1024, image.c_str(), cores, nullptr)) {
                    vms_[id] = std::move(vm);
                    response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nVM created with ID " + std::to_string(id);
                } else {
                    response = "HTTP/1.1 500 Internal Server Error\r\n\r\n";
                }
            }
        } else if (req.find("GET /vm/") == 0) {
            response = "HTTP/1.1 200 OK\r\n\r\nVM running";
        } else if (req.find("DELETE /vm/") == 0) {
            response = "HTTP/1.1 200 OK\r\n\r\nVM deleted";
        } else {
            response = "HTTP/1.1 404 Not Found\r\n\r\n";
        }
        send(client_fd, response.c_str(), response.length(), 0);
    }
};

#endif // ENABLE_HTTP
