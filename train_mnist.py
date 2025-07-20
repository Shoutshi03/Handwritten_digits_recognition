import tensorflow as tf
from mnist_model import create_cnn_model, load_and_preprocess_mnist

(train_images, train_labels), (test_images, test_labels) = load_and_preprocess_mnist()
input_shape = train_images.shape[1:]
num_classes = train_labels.shape[1]

model = create_cnn_model(input_shape, num_classes)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

model.save('mnist_cnn_model.h5')

print('Modèle entraîné et sauvegardé avec succès !')


