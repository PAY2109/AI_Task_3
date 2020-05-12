from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

number_of_experiment = '8'
#model_name = 'oneLayerModel'
model_name = 'twoLayerModel'





model = load_model(model_name)
img_rows = 28
img_cols = 28

print("Experiment №" + number_of_experiment + " for " + model_name)


image_path = 'experiment_images/experiment' + number_of_experiment + '/testimg2.png'
print("Prediction for " + image_path)
image = mpimg.imread(image_path)
plt.imshow(image.reshape(28, 28),cmap='Greys')
pred = model.predict(image.reshape(1, img_rows, img_cols, 1))
print(pred.argmax())
plt.show()

image_path = 'experiment_images/experiment' + number_of_experiment + '/testimg2_deviation1.png'
print("Prediction for " + image_path)
image = mpimg.imread(image_path)
plt.imshow(image.reshape(28, 28),cmap='Greys')
pred = model.predict(image.reshape(1, img_rows, img_cols, 1))
print(pred.argmax())
plt.show()

image_path = 'experiment_images/experiment' + number_of_experiment + '/testimg2_deviation2.png'
print("Prediction for " + image_path)
image = mpimg.imread(image_path)
plt.imshow(image.reshape(28, 28),cmap='Greys')
pred = model.predict(image.reshape(1, img_rows, img_cols, 1))
print(pred.argmax())
plt.show()

image_path = 'experiment_images/experiment' + number_of_experiment + '/testimg3.png'
print("Prediction for " + image_path)
image = mpimg.imread(image_path)
plt.imshow(image.reshape(28, 28),cmap='Greys')
pred = model.predict(image.reshape(1, img_rows, img_cols, 1))
print(pred.argmax())
plt.show()

image_path = 'experiment_images/experiment' + number_of_experiment + '/testimg3_deviation1.png'
print("Prediction for " + image_path)
image = mpimg.imread(image_path)
plt.imshow(image.reshape(28, 28),cmap='Greys')
pred = model.predict(image.reshape(1, img_rows, img_cols, 1))
print(pred.argmax())
plt.show()

image_path = 'experiment_images/experiment' + number_of_experiment + '/testimg3_deviation2.png'
print("Prediction for " + image_path)
image = mpimg.imread(image_path)
plt.imshow(image.reshape(28, 28),cmap='Greys')
pred = model.predict(image.reshape(1, img_rows, img_cols, 1))
print(pred.argmax())
plt.show()

image_path = 'experiment_images/experiment' + number_of_experiment + '/testimg8.png'
print("Prediction for " + image_path)
image = mpimg.imread(image_path)
plt.imshow(image.reshape(28, 28),cmap='Greys')
pred = model.predict(image.reshape(1, img_rows, img_cols, 1))
print(pred.argmax())
plt.show()

image_path = 'experiment_images/experiment' + number_of_experiment + '/testimg8_deviation1.png'
print("Prediction for " + image_path)
image = mpimg.imread(image_path)
plt.imshow(image.reshape(28, 28),cmap='Greys')
pred = model.predict(image.reshape(1, img_rows, img_cols, 1))
print(pred.argmax())
plt.show()

image_path = 'experiment_images/experiment' + number_of_experiment + '/testimg8_deviation2.png'
print("Prediction for " + image_path)
image = mpimg.imread(image_path)
plt.imshow(image.reshape(28, 28),cmap='Greys')
pred = model.predict(image.reshape(1, img_rows, img_cols, 1))
print(pred.argmax())
plt.show()

