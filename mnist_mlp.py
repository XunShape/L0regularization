import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os,sys,csv
mnist = input_data.read_data_sets("./data/MNIST/", one_hot=True)

np.set_printoptions(threshold=np.inf)
os.environ['CUDA_VISIBLE_DEVICES']='0'


def savew(fliename,output):
    data=[output]
    with open(fliename,'a') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(data)
    f.close()

def train(mnist,type,lamda,SIGMA,logdir):
    INPUT_NODE = 784     
    OUTPUT_NODE = 10     
    LAYER1_NODE = 400 
    LAYER2_NODE = 300 
    LAYER3_NODE = 100 

                                  
    BATCH_SIZE = 1000        

    LEARNING_RATE_BASE = 0.008      
    LEARNING_RATE_DECAY = 0.99    
    REGULARAZTION_RATE = 0.0001   
    TRAINING_STEPS = 3000        
    MOVING_AVERAGE_DECAY = 0.99 

    L0type1 = lambda V,SIGMA: tf.reduce_sum(tf.divide(tf.square(V),(tf.square(V)+tf.square(SIGMA))))
    L0type2 = lambda V,SIGMA: tf.reduce_sum(1-tf.exp(tf.divide(tf.square(V),(-2*tf.square(SIGMA)))))
    L0type3 = lambda V,SIGMA: tf.reduce_sum(tf.abs(tf.divide(1-tf.exp(-2*V),1+tf.exp(-2*V))))##(1-exp(-2*x)/1+exp(-2*x))

    L1 = lambda V: tf.reduce_sum(tf.abs(V))

    L2 = lambda V: tf.reduce_sum(tf.square(V))

    group = lambda V: tf.reduce_sum(tf.sqrt(tf.cast(V.get_shape().as_list()[1],tf.float32))*tf.sqrt(tf.reduce_sum(tf.square(V), axis=1)))
    
    def tf_round(x, decimals = 3):
        multiplier = tf.constant(10**decimals, dtype=x.dtype)
        return tf.round(x * multiplier) / multiplier
        
    def gpl1(v,SIGMA):
        return tf.reduce_sum([group(v)])+tf.reduce_sum([L1(v)])
        
        
    def L0_t1(v,SIGMA):
        return tf.reduce_sum([L0type1(v,SIGMA)])

    def L0_t2(v,SIGMA):
        return tf.reduce_sum([L0type2(v,SIGMA)])
        
    def L0_t3(v,SIGMA):
        return tf.reduce_sum([L0type3(v,SIGMA)])
        
    # def L20_t1
        
    def inference(input_tensor, avg_class, W, B):
        # no
        if avg_class == None:
            layer1 = tf.nn.relu(tf.matmul(input_tensor, W[0]) + B[0])
            layer2 = tf.nn.relu(tf.matmul(layer1, W[1]) + B[1])
            layer3 = tf.nn.relu(tf.matmul(layer2, W[2]) + B[2])
            return tf.matmul(layer3, W[3]) + B[3]
        
        else:
            
            layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(W[0])) + avg_class.average(B[0]))
            layer2 = tf.nn.relu(tf.matmul(layer1, avg_class.average(W[1])) + avg_class.average(B[1]))
            layer3 = tf.nn.relu(tf.matmul(layer2, avg_class.average(W[2])) + avg_class.average(B[2]))
            return tf.matmul(layer3, avg_class.average(W[3])) + avg_class.average(B[3])  
        
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, LAYER2_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[ LAYER2_NODE]))
    
    weights3 = tf.Variable(tf.truncated_normal([ LAYER2_NODE,  LAYER3_NODE], stddev=0.1))
    biases3 = tf.Variable(tf.constant(0.1, shape=[LAYER3_NODE]))
    
    weights4 = tf.Variable(tf.truncated_normal([LAYER3_NODE, OUTPUT_NODE], stddev=0.1))
    biases4 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    
    W=[weights1, weights2, weights3, weights4]
    B=[biases1, biases2, biases3, biases4]
    
    #HOOOOLY SHIT
    count_neurons = lambda W: tf.reduce_sum(tf.count_nonzero(tf.count_nonzero(tf_round(W, decimals = 3), 1)))
    sp_neurons = lambda W: tf.reduce_sum(tf.count_nonzero(tf_round(W, decimals = 3)))
    
    neurons1_summary = tf.reduce_sum([count_neurons(W[0])])
    neurons2_summary = tf.reduce_sum([count_neurons(W[1])])
    neurons3_summary = tf.reduce_sum([count_neurons(W[2])])
    neurons4_summary = tf.reduce_sum([count_neurons(W[3])])

    sp1_summary =  1-(tf.reduce_sum([sp_neurons(W[0])])/(INPUT_NODE*LAYER1_NODE))
    sp2_summary =  1-(tf.reduce_sum([sp_neurons(W[1])])/(LAYER2_NODE*LAYER1_NODE))
    sp3_summary =  1-(tf.reduce_sum([sp_neurons(W[2])])/(LAYER2_NODE*LAYER3_NODE))
    sp4_summary =  1-(tf.reduce_sum([sp_neurons(W[3])])/(LAYER3_NODE*OUTPUT_NODE))
    
    # non-average
    y = inference(x, None, W, B)
    
    # batch_ave
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, W, B)
    
    # cross_entropy
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    #reg_loss
    #regularizer = L0_t1(v,SIGMA)
    #regularizer = tf.contrib.layers.apply_regularization(L0_t1(W[0],SIGMA))
    if type=="l0t1":
        regularaztion = L0_t1(W[0],SIGMA)+L0_t1(W[1],SIGMA)+L0_t1(W[2],SIGMA)+L0_t1(W[3],SIGMA)
    elif type=="l0t2":
        regularaztion = L0_t2(W[0],SIGMA)+L0_t2(W[1],SIGMA)+L0_t2(W[2],SIGMA)+L0_t2(W[3],SIGMA)
    elif type=="l0t3":
        regularaztion = L0_t3(W[0],SIGMA)+L0_t3(W[1],SIGMA)+L0_t3(W[2],SIGMA)+L0_t3(W[3],SIGMA)
    elif type=="gpl":
        regularaztion = gpl1(W[0],SIGMA)+gpl1(W[1],SIGMA)+gpl1(W[2],SIGMA)+gpl1(W[3],SIGMA)

    #lr-log-loss
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    
    #summary
    with tf.name_scope('loss'):
        loss = cross_entropy_mean + regularaztion*lamda
        loss_summary = tf.summary.scalar('loss', loss)
        # reg_loss_summary = tf.summary.scalar('reg_loss', regularaztion)
        
        correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        TrAccuracy = tf.summary.scalar('TrAccuracy', accuracy)
    with tf.name_scope('spares_neurons'):
        neurons_layer1 = tf.summary.scalar('neurons_layer1', neurons1_summary)
        neurons_layer2 = tf.summary.scalar('neurons_layer2', neurons2_summary)
        neurons_layer3 = tf.summary.scalar('neurons_layer3', neurons3_summary)
        neurons_layer4 = tf.summary.scalar('neurons_layer4', neurons4_summary)
    with tf.name_scope('spares_weight'):
        weight_layer1 = tf.summary.scalar('weight_layer1',sp1_summary)
        weight_layer2 = tf.summary.scalar('weight_layer2',sp2_summary)
        weight_layer3 = tf.summary.scalar('weight_layer3',sp3_summary)
        weight_layer4 = tf.summary.scalar('weight_layer4',sp4_summary)
    
        # loss
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # train_step = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)
    #set mini-batch
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
        
    merged = tf.summary.merge([loss_summary,TrAccuracy,neurons_layer1,neurons_layer2,neurons_layer3,neurons_layer4,weight_layer1,weight_layer2,weight_layer3,weight_layer4])
    # start a session
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
        
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels} 
        #print(mnist.validation.labels)
        # train
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))
            
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            _,summary=sess.run([train_op,merged],feed_dict={x:xs,y_:ys})
            train_writer.add_summary(summary, i)

        test_acc,N1,N2,N3,N4,SP1,SP2,SP3,SP4 = sess.run([accuracy,neurons1_summary,neurons2_summary,neurons3_summary,neurons4_summary,sp1_summary,sp2_summary,sp3_summary,sp4_summary],feed_dict=test_feed)
        
        print(("After %d training step(s), test accuracy using average model is %g" %(TRAINING_STEPS, test_acc)))
        print(N1,N2,N3,N4,SP1,SP2,SP3,SP4)
        train_writer.flush()
        train_writer.close()
        return [N1,N2,N3,N4],[SP1,SP2,SP3,SP4],test_acc

try:
    type = sys.argv[1]       # 'l0t1', 'l0t2', 'l0t3', 'gpl'
except:
    print('Defaulted to group lasso l1 since no specified through command line')
    type = 'gpl'
    
if type=='gpl':
    logdir = "mnist/gpl/"
    lamda = 0.0001
    SIGMA = 0.0001
    train(mnist,type,lamda,SIGMA,logdir)
elif type=='l0t1':
    for lamda in [1000,100,10,1,0.1,0.01,0.001,0.0001]:
        for SIGMA in [0.01,0.001,0.0001]:
            logdir = "mnist/l0t1/lamda_"+str(lamda)+"_SIGMA_"+str(SIGMA)+"/"
            train(mnist,type,lamda,SIGMA,logdir)
elif type=='l0t2':
    for lamda in [1000,100,10,1,0.1,0.01,0.001,0.0001]:
        for SIGMA in [0.01,0.001,0.0001]:
            logdir = "mnist/l0t2/lamda_"+str(lamda)+"_SIGMA_"+str(SIGMA)+"/"
            train(mnist,type,lamda,SIGMA,logdir)
elif type=='l0t3':
    for lamda in [1000,100,10,1,0.1,0.01,0.001,0.0001]:
        for SIGMA in [0.01,0.001,0.0001]:
            logdir = "mnist/l0t3/lamda_"+str(lamda)+"_SIGMA_"+str(SIGMA)+"/"
            train(mnist,type,lamda,SIGMA,logdir)
elif type=='all':
    for lamda in [1000,100,10,1,0.1,0.01,0.001,0.0001]:
        for SIGMA in [0.01,0.001,0.0001]:
            logdir = "mnist/l0t1/lamda_"+str(lamda)+"_SIGMA_"+str(SIGMA)+"/"
            n,sp,teacc = train(mnist,"l0t1",lamda,SIGMA*5,logdir)
            fliename = "mnist/l0t1output.csv"
            output = [lamda,SIGMA*5,n,sp,teacc]
            savew(fliename,output)
    for lamda in [1000,100,10,1,0.1,0.01,0.001,0.0001]:
        for SIGMA in [0.01,0.001,0.0001]:
            logdir = "mnist/l0t2/lamda_"+str(lamda)+"_SIGMA_"+str(SIGMA)+"/"
            n,sp,teacc = train(mnist,"l0t2",lamda,SIGMA*5,logdir)
            fliename = "mnist/l0t2output.csv"
            output = [lamda,SIGMA*5,n,sp,teacc]
            savew(fliename,output)
    for lamda in [1000,100,10,1,0.1,0.01,0.001,0.0001]:
        for SIGMA in [0.01,0.001,0.0001]:
            logdir = "mnist/l0t3/lamda_"+str(lamda)+"_SIGMA_"+str(SIGMA)+"/"
            n,sp,teacc = train(mnist,"l0t3",lamda,SIGMA*5,logdir)
            fliename = "mnist/l0t3output.csv"
            output = [lamda,SIGMA*5,n,sp,teacc]
            savew(fliename,output)