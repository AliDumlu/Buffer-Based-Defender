# Import required libraries
from typing import Union
from art.estimators.classification import KerasClassifier
from art.attacks import evasion as attacks
import numpy as np
from PIL import Image
import datetime

KERAS_MODEL_TYPE = Union["keras.models.Model", "tf.keras.models.Model"]


class AdversarialGenerator:
    def __init__(self, n, x, path, classifier, image_classifier: KERAS_MODEL_TYPE):
        self.n = n
        self.x = x
        self.x_org = np.copy(x)
        self.path = path
        self.classifier = classifier
        self.image_classifier = image_classifier
        # A counter to monitor number of generated adversarial examples
        self.generation_counter = 0
        # Lists to store generated adversarial and original corresponding images and their file names
        self.x_adv = []
        # Predict the probabilities and classes of original examples
        self.org_prediction = image_classifier.predict(x)
        self.org_prediction_classes = np.argmax(self.org_prediction, axis=-1)
        self.adv_prediction = []
        self.adv_prediction_classes = []
        self.attack_type = ""

    # A function to prepare the path to save the image
    def prepare_path(self, typ, prediction, clas, counter):
        path = self.path + '\\' + self.attack_type + '\\' + self.classifier + '\\' + typ + '\\' + str(counter) \
               + '_' + str(clas) + '_' + str(np.max(prediction, axis=-1)) + '.png'
        return path

    def save_images(self, index):
        org_image = Image.fromarray(np.uint8(self.x[index] * 255))
        org_name = self.prepare_path('org', self.org_prediction[index], self.org_prediction_classes[index],
                                     self.generation_counter)
        org_image.save(org_name)
        # Save the Adversarial example
        adv_image = Image.fromarray(np.uint8(self.x_adv[index] * 255))
        adv_name = self.prepare_path('adv', self.adv_prediction[index], self.adv_prediction_classes[index],
                                     self.generation_counter)
        adv_image.save(adv_name)

    def delete_generated(self, index):
        self.x = np.delete(self.x, index, 0)
        self.org_prediction = np.delete(self.org_prediction, index, 0)
        self.org_prediction_classes = np.delete(self.org_prediction_classes, index, 0)
        self.x_adv = np.delete(self.x_adv, index, 0)
        self.adv_prediction = np.delete(self.adv_prediction, index, 0)
        self.adv_prediction_classes = np.delete(self.adv_prediction_classes, index, 0)
        self.generation_counter = self.generation_counter + 1

    def generate_adversarial_examples(self, attack_type):
        self.attack_type = attack_type
        # Prepare the wrapper for importing Keras model
        classifier = KerasClassifier(model=self.image_classifier)
        # Initialize the attack
        if attack_type == "FGSM":
            attack = attacks.FastGradientMethod(estimator=classifier, eps=0.007, eps_step=0.001,
                                                minimal=True, norm="inf")
        elif attack_type == "BIM":
            attack = attacks.BasicIterativeMethod(estimator=classifier, eps=0.007, eps_step=0.001, verbose=False)
        elif attack_type == "UAP":
            attack = attacks.UniversalPerturbation(classifier=classifier, attacker="fgsm",
                                                   attacker_params={"estimator":classifier, "eps":0.007, "eps_step":0.001, "minimal":True},
                                                   eps=0.07)
        elif attack_type == "ST":
            attack = attacks.SpatialTransformation(classifier=classifier, max_translation=10, max_rotation=15,
                                                   num_translations=5, num_rotations=5)
        elif attack_type == "ZOO":
            attack = attacks.ZooAttack(classifier=classifier, abort_early=True)
        else:
            attack = attacks.PixelAttack(classifier=classifier, th=1, es=1, targeted=False, verbose=False)

        while self.generation_counter < self.n:
            print(str(self.generation_counter) + " | " + str(datetime.datetime.now()))
            # Generate adversarial examples
            self.x_adv = attack.generate(x=self.x)
            # Predict the probabilities and classes of the adversarial examples
            self.adv_prediction = self.image_classifier.predict(self.x_adv)
            self.adv_prediction_classes = np.argmax(self.adv_prediction, axis=-1)
            i = 0
            j = 0
            while i < len(self.x_adv):
                index = i - j
                # Check if the example is classified correctly and if the adversarial example fooled the model
                if self.adv_prediction_classes[index] != self.org_prediction_classes[index]:
                    self.save_images(index)
                    self.delete_generated(index)
                    j = j + 1
                i = i + 1
        self.x = np.copy(self.x_org)
