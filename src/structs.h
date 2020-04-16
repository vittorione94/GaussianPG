#ifndef MYSTRUCT_H
#define MYSTRUCT_H
#endif

struct Policy: torch::nn::Module {
    bool softplus = false;
    int HID_SIZE = 128;

    Policy(int obs_size, int action_size, bool softplus_param = false) {
        softplus = softplus_param;
        fc1 = register_module("fc1", torch::nn::Linear(obs_size, HID_SIZE));
        fc2 = register_module("fc2", torch::nn::Linear(HID_SIZE, HID_SIZE));
        fc3 = register_module("fc3", torch::nn::Linear(HID_SIZE, action_size));
    }

    // Implement Algorithm
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::tanh(fc2->forward(x));

        if (softplus)
            x = torch::softplus(fc3->forward(x));
        else
            x = torch::tanh(fc3->forward(x));

        return x;
    }

    // Declare layers
    torch::nn::Linear fc1{nullptr}, fc2{nullptr},fc3{nullptr};
};

struct ValueFn: torch::nn::Module {
    int HID_SIZE = 80;
    ValueFn(int obs_size, int action_size) {
        fc1 = register_module("fc1", torch::nn::Linear(obs_size, HID_SIZE));
        fc2 = register_module("fc2", torch::nn::Linear(HID_SIZE, HID_SIZE));
        fc3 = register_module("fc3", torch::nn::Linear(HID_SIZE, action_size));
    }
    // Implement Algorithm
    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(fc1->forward(x));
        x = torch::tanh(fc2->forward(x));
        x = torch::tanh(fc3->forward(x));
        return x;
    }
    torch::nn::Linear fc1{nullptr}, fc2{nullptr},fc3{nullptr};
};

