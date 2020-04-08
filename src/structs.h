#ifndef MYSTRUCT_H
#define MYSTRUCT_H
#endif

struct Policy: torch::nn::Module {

    Policy() {
        fc1 = register_module("fc1", torch::nn::Linear(3, 50));
        fc2 = register_module("fc2", torch::nn::Linear(50, 20));
        fc3 = register_module("fc3", torch::nn::Linear(20, 1));
    }

    // Implement Algorithm
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }

    // Declare layers
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

struct ValueFn: torch::nn::Module {
    ValueFn() {
        fc1 = register_module("fc1", torch::nn::Linear(3, 50));
        fc3 = register_module("fc3", torch::nn::Linear(50, 1));
    }

    // Implement Algorithm
    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = fc3->forward(x);
        return x;
    }

    // Declare layers fc3{nullptr}
    torch::nn::Linear fc1{nullptr}, fc3{nullptr};
};

