from tensorflow.keras.models import load_model
model = load_model('twoLayerModel')

#image_index = 4444
img_rows = 28
img_cols = 28

image_path = 'testimg8.png'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread(image_path)

#plt.imshow(image)

plt.imshow(image.reshape(28, 28),cmap='Greys')
pred = model.predict(image.reshape(1, img_rows, img_cols, 1))
print(pred.argmax())
plt.show()
