import tensorflow as tf
import neural_net as N
import input_data as D
import matplotlib.pyplot as plt
input=tf.compat.v1.placeholder(tf.float32,[None,2],name="input")
input_x=tf.compat.v1.placeholder(tf.float32,[None,1],name="input_x")
potential=input_x**2#谐振子
n_layers=1
n_neurons=10
output_dim=1
learning_rate=0.01
epoch_num=100000


sess=tf.compat.v1.Session()
net=N.build_w_l_s(input,potential,n_layers,n_neurons,output_dim)

opt=tf.train.AdadeltaOptimizer(learning_rate)
train_opt=opt.minimize(net.loss)
sess.run(tf.global_variables_initializer())

train_loss_lst=[]
E_i_lst=[]
E_r_lst=[]
for i in range(epoch_num):
    input_g,input_x_g=D.generate()

    feed_dict={input:input_g,input_x:input_x_g}
    loss,_,E_i,E_r=sess.run([net.loss,train_opt,net.E_i,net.E_r],feed_dict)
    if i%100==0:
        mes="loss is %s,i is %s,E_r is %s, E_i is %s"%(loss,i,E_r,E_i)
        print(mes)
        if i%100==0:
            train_loss_lst.append(loss)
            E_i_lst.append(E_i)
            E_r_lst.append(E_r)
#plt.plot(train_loss_lst)
plt.plot(E_i_lst)
plt.plot(E_r_lst)
plt.show()
