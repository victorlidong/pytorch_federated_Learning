"""
Example membership inference attack against a deep net classifier on the CIFAR10 dataset
"""
import sys
sys.path.append('/media/aaa/041CDACD1CDAB93E/pyProject/mia')
from dp import dp_utils
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential, metrics
from sklearn.model_selection import train_test_split
from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data
from sklearn.metrics import roc_curve
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--is_log', type=bool, default=False)
parser.add_argument('--log_path', type=str, default="tmp.log")
parser.add_argument('--is_dp', type=bool, default=False)
parser.add_argument('--pb', type=float, default=1e6)
parser.add_argument('--clip_bound', type=float, default=0.1)
parser.add_argument('--dp_type', type=str,default="norm1")
parser.add_argument('--target_epochs',type=int,default=12,help="Number of epochs to train target and shadow models")
parser.add_argument('--attack_epochs',type=int,default=12,help="Number of epochs to train attack models")
parser.add_argument('--num_shadows',type=int,default=3,help="num_shadows")

args = parser.parse_args()


NUM_CLASSES = 10
WIDTH = 32
HEIGHT = 32
CHANNELS = 3
SHADOW_DATASET_SIZE = 4000
ATTACK_TEST_DATASET_SIZE = 4000



# log
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def get_data():
    """Prepare CIFAR10 data."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    X_train /= 255
    X_test /= 255

    # train_size_num=20000
    train_size_num = 5000
    #把训练集缩小
    X_train, tmp_x_test, y_train, tmp_y_test = train_test_split(X_train, y_train, test_size=(1-train_size_num/50000.0),random_state=1)
    return (X_train, y_train), (X_test, y_test)


def target_model_fn():
    """The architecture of the target (victim) model.

    The attack is white-box, hence the attacker is assumed to know this architecture too."""

    model = tf.keras.models.Sequential()

    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=(WIDTH, HEIGHT, CHANNELS),
        )
    )
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def attack_model_fn():
    """Attack model that takes target model predictions and predicts membership.

    Following the original paper, this attack model is specific to the class of the input.
    AttachModelBundle creates multiple instances of this model for each class.
    """
    model = tf.keras.models.Sequential()

    model.add(layers.Dense(128, activation="relu", input_shape=(NUM_CLASSES,)))

    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))

    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_target_model(model,X,Y,epochs=12,is_dp=False,dp_type="norm1",privacy_budget=1e6,clip_bound=0.1,sample_num=500,privacy_delta=1e-6,parallelnum=1):
    batch_size = 32
    x_train, x_test, y_train, y_test = train_test_split(X, Y,test_size=0.1,random_state=1)

    print("train size: ",x_train.shape,", val size: ",x_test.shape)
    if is_dp:
        print("use dp,type=", dp_type)

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    val_dataset=tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)


    optimizer = optimizers.Adam(learning_rate=0.0005)  # 声明采用批量随机梯度下降方法，学习率=0.01
    acc_meter = metrics.Accuracy()
    val_acc_meter = metrics.Accuracy()


    iteration=0
    for epoch in range(1,epochs+1):

        for step, (x, y) in enumerate(dataset):  # 一次输入batch组数据进行训练
            iteration+=1
            with tf.GradientTape() as tape:  # 构建梯度记录环境
                out = model(x)
                loss = tf.square(out - y)
                loss = tf.reduce_sum(loss) / batch_size #定义均方差损失函数
                grads = tape.gradient(loss, model.trainable_variables)  # 计算网络中各个参数的梯度

            #add noise

            if is_dp:
                if dp_type == "norm1":
                    tensor_size_all = 0
                    for grad in grads:
                        tensor_size_all += dp_utils.get_tensor_size(grad.shape.dims)
                    for i, grad in enumerate(grads):
                        grad = dp_utils.clip_func(clip_bound, dp_type, grad)
                        sensitivity = dp_utils.calculate_l1_sensitivity(clip_bound, tensor_size_all)
                        beta = dp_utils.gen_laplace_beta(batch_size, parallelnum, sensitivity, privacy_budget)
                        noise_tensor = tf.cast(tf.convert_to_tensor(dp_utils.laplace_function(beta, grad.shape.dims)),
                                               dtype=tf.float32)
                        grads[i]+=noise_tensor


                elif dp_type == "norm2":
                    for i, grad in enumerate(grads):
                        grad = dp_utils.clip_func(clip_bound, dp_type, grad)
                        sensitivity = dp_utils.calculate_l2_sensitivity(clip_bound)
                        sigma = dp_utils.gen_gaussian_sigma(batch_size, parallelnum, sensitivity, privacy_budget, privacy_delta)
                        noise_tensor=tf.random.normal(grad.shape.dims, stddev=sigma, dtype=tf.float32)
                        grads[i] += noise_tensor


                elif dp_type == "sample_L1":
                    for i, grad in enumerate(grads):
                        tensor_size = dp_utils.get_tensor_size(grad.shape.dims)
                        sensitivity = dp_utils.calculate_l1_sensitivity_sample(grad, tensor_size, sample_num)
                        beta = dp_utils.gen_laplace_beta(batch_size, parallelnum, sensitivity, privacy_budget)
                        noise_tensor = tf.cast(tf.convert_to_tensor(dp_utils.laplace_function(beta, grad.shape.dims)),
                                               dtype=tf.float32)
                        grads[i] += noise_tensor


                elif dp_type == "sample_L2":
                    for i, grad in enumerate(grads):
                        tensor_size = dp_utils.get_tensor_size(grad.shape.dims)
                        sensitivity = dp_utils.calculate_l2_sensitivity_sample(grad, tensor_size, sample_num)
                        sigma = dp_utils.gen_gaussian_sigma(batch_size, parallelnum, sensitivity, privacy_budget, privacy_delta)
                        noise_tensor = tf.random.normal(grad.shape.dims, stddev=sigma, dtype=tf.float32)
                        grads[i] += noise_tensor



            #add noise done
            with tf.GradientTape() as tape:  # 构建梯度记录环境
                optimizer.apply_gradients(zip(grads, model.trainable_variables))  # 更新网络参数
                acc_meter.update_state(tf.argmax(out, axis=1), tf.argmax(y, axis=1))  # 比较预测值与标签，并计算精确度



            if iteration % 100 == 0:  #
                print('Epoch',epoch,'iteration', iteration, ': Loss is: ', float(loss), ' Train Accuracy: ', acc_meter.result().numpy())
                acc_meter.reset_states()

        #每一个epoch验证一次

        for step, (x, y) in enumerate(val_dataset):
            out = model(x)
            prediction=tf.argmax(out, axis=1)
            label=tf.argmax(y, axis=1)
            val_acc_meter.update_state(prediction,label)
        print('Epoch', epoch, 'iteration', iteration, ' Val Accuracy: ', val_acc_meter.result().numpy())
        val_acc_meter.reset_states()



def demo():


    (X_train, y_train), (X_test, y_test) = get_data()

    # Train the target model.
    print("Training the target model...")
    target_model = target_model_fn()
    print("target model")
    print(target_model.summary())


    # target_model训练
    train_target_model(target_model,X_train,y_train,epochs=args.target_epochs,
                       is_dp=args.is_dp,dp_type=args.dp_type,privacy_budget=args.pb,clip_bound=args.clip_bound)

    # target_model.fit(
    #     X_train, y_train, epochs=12, validation_split=0.1, verbose=True
    # )
    print("Training the target model... done!!!")

#-----------------------------------------------------------------------------------

    # Train the shadow models.
    smb = ShadowModelBundle(
        target_model_fn,
        shadow_dataset_size=SHADOW_DATASET_SIZE,
        num_models=args.num_shadows,
    )

    #把测试集数据按照9:1分成影子模型的训练集和测试集
    # We assume that attacker's data were not seen in target's training.
    attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
        X_test, y_test, test_size=0.1
    )
    print(attacker_X_train.shape, attacker_X_test.shape)

    print("Training the shadow models...")
    X_shadow, y_shadow = smb.fit_transform(
        attacker_X_train,
        attacker_y_train,
        fit_kwargs=dict(
            epochs=args.target_epochs,
            verbose=True,
            validation_data=(attacker_X_test, attacker_y_test),
        ),
    )

    # ShadowModelBundle returns data in the format suitable for the AttackModelBundle.
    amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)

    # Fit the attack models.
    print("Training the attack models...")
    amb.fit(
        X_shadow, y_shadow, fit_kwargs=dict(epochs=args.attack_epochs, verbose=True)
    )

    # Test the success of the attack.

    # Prepare examples that were in the training, and out of the training.
    data_in = X_train[:ATTACK_TEST_DATASET_SIZE], y_train[:ATTACK_TEST_DATASET_SIZE]
    data_out = X_test[:ATTACK_TEST_DATASET_SIZE], y_test[:ATTACK_TEST_DATASET_SIZE]

    # Compile them into the expected format for the AttackModelBundle.
    attack_test_data, real_membership_labels = prepare_attack_data(
        target_model, data_in, data_out
    )

    # Compute the attack accuracy.



    attack_guesses = amb.predict(attack_test_data)
    attack_accuracy = np.mean(attack_guesses == real_membership_labels)
    fpr, tpr, phi = roc_curve(real_membership_labels, attack_guesses, pos_label=1)
    Adv_A = tpr - fpr
    print("attack_accuracy=",attack_accuracy)
    print("Privacy Leakage Metrics=",Adv_A)


if __name__ == "__main__":

    log_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 控制台输出log重定向
    if args.is_log:
        sys.stdout = Logger(args.log_path + '-' + log_time + '.txt', sys.stdout)
        sys.stderr = Logger(args.log_path + '-' + log_time + '.txt', sys.stderr)

    demo()
