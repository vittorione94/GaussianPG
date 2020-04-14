#include <torch/torch.h>

#include <iostream>
#include <chrono>  // for high_resolution_clock
using namespace torch;
using namespace std;
#include "structs.h"
#include <random>
#include <math.h>

class A2CAgent
{
  public:
    Policy policy_mu;
    ValueFn value_fn;
    at::Tensor dist;

    at::Tensor mu;
    double sigma = 0.1;
    A2CAgent()
    {
      this->policy_mu = Policy();
      this->value_fn = ValueFn();
    }

    ~A2CAgent()
    {
    }

    at::Tensor logpdf(double x, double mu, double sigma)
    {
        auto t = at::tensor({2*M_PI*sigma*sigma});
        return -0.5 * (x-mu) * (x-mu)/(sigma*sigma) - 0.5 * torch::log(t);
    }

    double normpdf(double x, double mu, double sigma)
    {
        //std::cout << "---normpdf --- " << (1/ (sigma * std::sqrt(2*M_PI)) ) * std::exp(- std::pow(x-mu, 2.0) / (2*sigma*sigma)) <<  std::endl;
        return (1/ (sigma * std::sqrt(2*M_PI)) ) * std::exp(- std::pow(x-mu, 2.0) / (2*sigma*sigma));
    }

    torch::Tensor grad_norm(double x, double mu, double sigma, torch::Tensor state)
    {
        return ( (1/ (sigma * sigma )) * (x - mu) * state);
    }

    double act(torch::Tensor state)
    {
          torch::Tensor out = policy_mu.forward(state);
          this->mu = out[0];
          auto action = torch::normal(this->mu.item<double>(), sigma, {1,1});
          action = torch::clamp(action, -2.0, 2.0);

          return action.item<double>();
    }

    void learn(torch::Tensor state,
               torch::Tensor next_state,
               double reward,
               double action,
               torch::optim::Adam mu_optim,
               torch::optim::Adam value_optim,
               double gamma = 0.9
               )
    {
      torch::Tensor state_value = value_fn.forward(state);
      torch::Tensor next_state_value = value_fn.forward(next_state);
      auto value_loss = torch::pow((reward + gamma*next_state_value) - state_value, 2);

      //std::cout<< "loss : " << loss <<std::endl;
      value_optim.zero_grad();
      value_loss.backward();
      value_optim.step();

      state_value = value_fn.forward(state).detach();
      next_state_value = value_fn.forward(next_state).detach();

      auto log_unnormalized = -0.5 * torch::pow((action / this->sigma) - (this->mu / this->sigma) ,2);
      auto log_normalization = 0.5 * torch::log(torch::tensor({2. * M_PI})) + torch::log(torch::tensor({this->sigma}));
      auto log_prob = log_unnormalized - log_normalization;

      auto loss = -log_prob * ((reward + gamma*next_state_value) - state_value);
      //loss -= 1e-2 * std::log(4); // to encourage exploration
        std::cout << "#### reward + gamma*next_state_value - state_value " << (reward + gamma*next_state_value) - state_value << std::endl;
        std::cout << "#### log prob " << log_prob << std::endl;
        std::cout << "#### loss " << loss << std::endl;
      //auto loss = -this->dist.log_prob(action) * ((reward + gamma*next_state_value) - state_value);

      policy_mu.zero_grad();
      //sigma_optim.zero_grad();
      loss.backward();
      for (auto& p : policy_mu.parameters())
      {
        //std::cout << "----- p grad " << p.grad() << std::endl;
        auto tmp = 0.001*p.grad();
        p = p.detach();
        p -= tmp;
      }
      //mu_optim.step();
      //sigma_optim.step();
    }
};
