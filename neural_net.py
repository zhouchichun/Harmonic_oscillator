import sys
import numpy as np
import tensorflow as tf
# neural_net 构建波函数,拉氏量，作用量

class build_w_l_s():
    def __init__(self,inputs,potential,n_layers,n_neurons,output_dim):
        self.inputs=inputs
        self.n_layers=n_layers
        self.n_neurons=n_neurons
        self.output_dim=output_dim
        self.potential=potential
        
        self.build_wave_function()
        self.build_dirive()
        
        self.build_norm()
        self.build_laplace()
        self.build_action()
        self.loss()
        self.build_E_i()
        self.build_E_r()
        #exit()
       # self.initialize()
    def build_wave_function(self):
        #定义实部
        out_tmp_r=tf.layers.dense(self.inputs,self.n_neurons,activation=tf.nn.tanh)
        for i in range(1,self.n_layers):
            out_tmp_r=tf.layers.dense(out_tmp_r,self.n_neurons,activation=tf.nn.tanh)
        self.output_r=tf.layers.dense(out_tmp_r,self.output_dim)
        #self.u=tf.squeeze(self.output_r)
        self.u=self.output_r
        print(self.u)
        print("-----------------------------")
        #定义虚部
        out_tmp_i=tf.layers.dense(self.inputs,self.n_neurons,activation=tf.nn.tanh)
        for i in range(1,self.n_layers):
            out_tmp_i=tf.layers.dense(out_tmp_i,self.n_neurons,activation=tf.nn.tanh)
        self.output_i=tf.layers.dense(out_tmp_i,self.output_dim)
        #self.v=tf.squeeze(self.output_i)
        self.v=self.output_i
        print(self.v)
        print("-----------------------------")
        
    def build_dirive(self):
        self.v_d_lst=tf.gradients(self.v,self.inputs)
        self.u_d_lst=tf.gradients(self.u,self.inputs)
        
        
    def build_norm(self):
        self.norm=self.u*self.u+self.v*self.v
        self.norm =tf.reduce_sum(self.norm)
        
    def build_laplace(self):
        self.laplace=self.u*self.v_d_lst[0]-self.v*self.u_d_lst[0]
        for u_d,v_d in zip(self.u_d_lst[1:],self.v_d_lst[1:]):
            self.laplace +=-0.5*u_d*u_d-0.5*v_d*v_d
        self.laplace=self.potential*self.u*self.u+self.potential*self.v*self.v
        print(self.laplace)
        
    def build_action(self):
        self.action =tf.reduce_sum(self.laplace)
    def loss(self):
        self.loss=self.action+(self.norm-1)**2
        print("lossssssssss")
        print(self.loss)
    def build_E_i(self):
        self.E_i_each=self.u*self.u_d_lst[0]+self.v*self.v_d_lst[0]
        self.E_i=tf.reduce_sum(self.E_i_each)
        print(self.E_i)
    def build_E_r(self):
        self.E_r_each=self.v*self.u_d_lst[0]-self.u*self.v_d_lst[0]
        self.E_r=tf.reduce_sum(self.E_r_each)
        print(self.E_r)
