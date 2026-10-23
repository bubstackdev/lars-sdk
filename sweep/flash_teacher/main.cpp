#include "session.hpp"
#include <iostream>

int main(int argc, char **argv) {
    try {
        bool mock = false, gpu = false;
        int context = 8192;
        std::string model;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--mock") mock = true;
            else if (a == "--allow-gpu") gpu = true;
            else if ((a == "--model" || a == "--context") && i + 1 < argc) {
                if (a == "--model") model = argv[++i]; else context = std::stoi(argv[++i]);
            } else throw std::invalid_argument("usage: flash-teacher --model GGUF [--allow-gpu] [--context N] | --mock");
        }
        if (context < 1 || context > 65536 || (mock && gpu)) throw std::invalid_argument("invalid context or conflicting modes");
        std::unique_ptr<teacher::Engine> engine;
        if (mock) engine = std::make_unique<teacher::MockEngine>();
        else {
#ifdef FLASH_CPU_ONLY
            throw std::invalid_argument("this CPU test executable requires --mock");
#else
            if (model.empty()) throw std::invalid_argument("--model is required");
            engine = teacher::runtime_engine(model, context, gpu);
#endif
        }
        teacher::Session session(*engine, context);
        auto ready = engine->info(); ready["protocol"] = 1; ready["context"] = context; ready["ok"] = true;
        ready["label_prefixes"] = {"", " "};
        ready["generation"] = "greedy-v1";
        std::cout << ready.dump() << std::endl;
        std::string line;
        while (std::getline(std::cin, line)) {
            teacher::json id = nullptr;
            try {
                if (line.size() > 16 * 1024 * 1024) throw std::invalid_argument("request too large");
                const auto request = teacher::json::parse(line);
                id = request.value("id", teacher::json(nullptr));
                auto result = session.dispatch(request);
                std::cout << teacher::json({{"id", id}, {"ok", true}, {"result", result}}).dump() << std::endl;
                if (request.at("op") == "quit") break;
            } catch (const teacher::Fatal &e) {
                std::cout << teacher::json({{"id", id}, {"ok", false}, {"fatal", true}, {"error", e.what()}}).dump() << std::endl;
                return 75;
            } catch (const std::exception &e) {
                session.clear();
                std::cout << teacher::json({{"id", id}, {"ok", false}, {"error", e.what()}}).dump() << std::endl;
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "flash-teacher: " << e.what() << '\n';
        return 1;
    }
}
