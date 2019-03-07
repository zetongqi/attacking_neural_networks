import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from keras.models import load_model
from keras import backend as K

# load 99.25% accuracy CNN for MNIST dataset
model = load_model('/Users/zetong/cnn.h5')
model.summary()

# helper functions
# load the image from MNIST dataset into [1,28,28,1] dimension
def load_image(image_location):
	x = np.asarray(PIL.Image.open(image_location))
	x = x.reshape(28,28)
	x = x.reshape(1, 28, 28, 1)
	x = x.astype('float32')
	x /= 255
	return x

# putting the adversarial examples and their misclassification labels in subplots
def plot_adversarials(attacks, predicted_labels):
	w=10
	h=10
	fig=plt.figure(figsize=(8, 8))
	columns = len(attacks)
	rows = 1
	for i in range(1, columns*rows+1):
		ax = fig.add_subplot(rows, columns, i)
		ax.title.set_text(str(predicted_labels[i-1]))
		plt.imshow(attacks[i-1].reshape((28,28)), cmap='Greys')
	plt.show()
	return

# a helper function to visualize a list of adversarial examples
def visualize_adversarials(attacks):
	w=10
	h=10
	fig=plt.figure(figsize=(8, 8))
	columns = len(attacks)
	rows = 1
	for i in range(1, columns*rows+1):
		ax = fig.add_subplot(rows, columns, i)
		plt.imshow(attacks[i-1].reshape((28,28)), cmap='Greys')
	plt.show()
	return

class cwl2:
	def __init__(self, sess, model, K=0, max_iterations=1000, step_size=0.1, initial_cost=10, SEARCH_STEPS=9):
		self.sess = sess
		self.model = model
		self.K = K
		self.max_iterations = max_iterations
		self.step_size = step_size
		self.initial_cost = initial_cost
		self.SEARCH_STEPS = SEARCH_STEPS
		self.num_classes = 10
    
	# return the logits tensor and the nontarget index logits tensor
	def get_logits_and_nontarget_logits(self, x_new, target):
		logits = self.model(x_new)
		return logits, tf.concat([logits[0][0:target], logits[0][target:-1]], 0)
    
	# the loss function for l2 attack
	def loss(self, x_new, x, target, c):
		logits, nontarget_logits = self.get_logits_and_nontarget_logits(x_new, target)
		return tf.norm(x_new-x)**2 + c*tf.math.maximum(tf.math.reduce_max(nontarget_logits) - logits[0][target], -self.K)
        
	# search for the best adversarial examples
	def find_best_attack(self, x, target, DEBUG=1):
		# all the valid attacks
		attacks = []
		# initial c value
		c = self.initial_cost
		# confidence parameter
		K = self.K
		# initialize all global variables
		self.sess.run(tf.global_variables_initializer())
		# for every search steps
		for s in range(self.SEARCH_STEPS):
			# the adversarial example the algorithm is searching for
			x_new = tf.Variable(np.zeros([1,28,28,1]), dtype=tf.float32)
			loss = self.loss(x_new, x, target, c)
			start_vars = set(x.name for x in tf.global_variables())
			train = tf.train.AdamOptimizer(self.step_size).minimize(loss, var_list=[x_new])
			end_vars = tf.global_variables()
			# variables in Adam optimizer
			new_vars = [x for x in end_vars if x.name not in start_vars]
			# initilize x_new and Adam optimizer variables
			self.sess.run(tf.variables_initializer(var_list=[x_new]+new_vars))

			for i in range(self.max_iterations):
				self.sess.run(train)
			new_img = self.sess.run(x_new)
			# if the attack is sucessful
			if np.argmax(self.model.predict(new_img)) == target:
				# add it to the valid attacks collection
				attacks.append(new_img)
				# decrease c value to try to find an attack with less perturbation
				c = c*0.5
			# if the attack failed
			else:
				# increase c value to enforce the solver to find a sucessful attack
				c = c*10
			# if debug mode, print out the each search step's detail
			if DEBUG == 1:
				print(s+1, "/", self.SEARCH_STEPS, "search steps")
				print("c value:", c)
				print(len(attacks), "adversarials examples found")
		# let the best attack to be the one with the smallest perturbation
		noise_norms = []
		for i in range(len(attacks)):
			noise_norms.append(np.linalg.norm(attacks[i] - x))
		best_attack = attacks[np.argmin(noise_norms)]
		return best_attack, attacks

	def untargeted_attack(self, x):
		attacks = []
		predicted_labels = []
		for i in range(self.num_classes):
			print("attacking target label", i)
			best_attack, all_attacks = self.find_best_attack(x, i, DEBUG=0)
			attacks.append(best_attack)
			predicted_labels.append(np.argmax(self.model.predict(best_attack)))
			print(len(all_attacks), "attacks found")
		plot_adversarials(attacks, predicted_labels)
		return attacks

# targeted attack test
with tf.Session() as sess:
	x = load_image('/Users/zetong/mnist_png/testing/5/53.png')
	l2attack = cwl2(sess, model)
	best_attack, attacks = l2attack.find_best_attack(x, 9)
	visualize_adversarials(attacks)