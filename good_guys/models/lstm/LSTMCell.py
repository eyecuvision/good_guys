import torch.nn as N


class LSTMCell(N.Module):


    def __init__(self,state_size = 128,input_size = 1024):
        super().__init__()

        self.state_size = state_size
        self.input_size = input_size


        self.forget_gate = N.Sequential(
            N.Linear(self.input_size,self.state_size),
            N.Sigmoid()
        )

        self.input_gate = N.Sequential(
            N.Linear(self.input_size,self.state_size),
            N.Sigmoid()
        )

        self.input_gate_tanh = N.Sequential(
            N.Linear(self.input_size,self.state_size),
            N.Tanh()
        )

        self.output = N.Sequential(
            N.Linear(self.state_size,1),
            N.Sigmoid()
        )

    def forward(self,s_prev,x_t):

        s_t = self.forget_gate(x_t) * s_prev
        s_t = s_t + self.input_gate(x_t) * self.input_gate_tanh(x_t)
        p_t = self.output(s_t)

        return s_t,p_t

