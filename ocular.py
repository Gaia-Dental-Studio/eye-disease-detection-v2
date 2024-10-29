import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras import backend as K



@st.cache_resource()
def load_model(path):
	model = tf.keras.models.load_model(path,compile=False)
	model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])
	return model


@tf.function
def accuracy_multilabel(y, y_hat):
    correct_prediction = tf.equal(tf.round(y_hat), y)
    correct_prediction = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return correct_prediction

class MetricsAtTopK:
    def __init__(self, k):
        self.k = k

    def _get_prediction_tensor(self, y_pred):
        """Takes y_pred and creates a tensor of same shape with 1 in indices where, the values are in top_k
        """
        topk_values, topk_indices = tf.nn.top_k(y_pred, k=self.k, sorted=False, name="topk")
        ii, _ = tf.meshgrid(tf.range(tf.shape(y_pred)[0]), tf.range(self.k), indexing='ij')
        index_tensor = tf.reshape(tf.stack([ii, topk_indices], axis=-1), shape=(-1, 2))
        prediction_tensor = y_pred
        prediction_tensor = tf.cast(prediction_tensor, K.floatx())
        return prediction_tensor

    def true_positives_at_k(self, y_true, y_pred):
        prediction_tensor = self._get_prediction_tensor(y_pred=y_pred)
        true_positive = K.sum(tf.multiply(prediction_tensor, y_true))
        return true_positive

    def false_positives_at_k(self, y_true, y_pred):
        prediction_tensor = self._get_prediction_tensor(y_pred=y_pred)
        true_positive = K.sum(tf.multiply(prediction_tensor, y_true))
        c2 = K.sum(prediction_tensor)  # TP + FP
        false_positive = c2 - true_positive
        return false_positive

    def false_negatives_at_k(self, y_true, y_pred):
        prediction_tensor = self._get_prediction_tensor(y_pred=y_pred)
        true_positive = K.sum(tf.multiply(prediction_tensor, y_true))
        c3 = K.sum(y_true)  # TP + FN
        false_negative = c3 - true_positive
        return false_negative

    def precision_at_k(self, y_true, y_pred):
        prediction_tensor = self._get_prediction_tensor(y_pred=y_pred)
        true_positive = K.sum(tf.multiply(prediction_tensor, y_true))
        c2 = K.sum(prediction_tensor)  # TP + FP
        return true_positive/(c2+K.epsilon())

    def recall_at_k(self, y_true, y_pred):
        prediction_tensor = self._get_prediction_tensor(y_pred=y_pred)
        true_positive = K.sum(tf.multiply(prediction_tensor, y_true))
        c3 = K.sum(y_true)  # TP + FN
        return true_positive/(c3+K.epsilon())

    def f1_at_k(self, y_true, y_pred):
        precision = self.precision_at_k(y_true=y_true, y_pred=y_pred)
        recall = self.recall_at_k(y_true=y_true, y_pred=y_pred)
        f1 = (2*precision*recall)/(precision+recall+K.epsilon())
        return f1

metricsAtTopK = MetricsAtTopK(k=5)
AUC_value = tf.keras.metrics.AUC( curve='ROC', summation_method='interpolation', multi_label=True)
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.00001)

@st.cache_resource()
def load_model_multiclass(path):
	model = tf.keras.models.load_model(path,compile=False)
	model.compile(loss="categorical_crossentropy",
              optimizer=OPTIMIZER,
              metrics=[accuracy_multilabel,
                       metricsAtTopK.f1_at_k,
                       AUC_value])
	return model




def predict_class(image, model):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [224, 224])

	image = np.expand_dims(image, axis = 0)

	prediction = model.predict(image)

	return prediction

model_multiclass = load_model_multiclass('models/final_model_20240122-033648_multiclass.h5')
model_cataract = load_model('models/final_model_20240119-154434_cataract_effnet.h5')
model_diabetes = load_model('models/final_model_20240119-172037_diabetes_effnet.h5')
model_glaucoma = load_model('models/final_model_20240119-174234_glaucoma_effnet.h5')
model_hypertension = load_model('models/final_model_20240122-074201_hypertension_effnet.h5')
model_other = load_model('models/final_model_20240122-122757_others_effnet.h5')
MODELS = [model_cataract,model_diabetes,model_glaucoma,model_hypertension,model_other]
DISEASE = ['cataract','diabetes','glaucoma','hypertension','other']
st.title('Ocular Disease Classifier')

MULTILABEL_CLASS = ['A','C','D','G','H','M','N','O']


st.sidebar.header("Model Selection")
classification_selection = st.sidebar.radio("Select model",options=["Full Model","Partial Model"])


file = st.file_uploader("Upload an eye image", type=["jpg", "png","jpeg"])


if file is None:
	st.text('Waiting for upload....')

else:
	slot = st.empty()
	slot.text('Running inference....')
	
	test_image = Image.open(file)
       
	if classification_selection=="Full Model":
		st.image(test_image, caption="Input Image", width = 400)
		pred = predict_class(np.asarray(test_image), model_multiclass)
		for i in range(len(MULTILABEL_CLASS)):
			st.write(f"Confidence of {MULTILABEL_CLASS[i]}: {pred[0][i]}")
		st.divider()
		st.write(f"Best Result : {str(MULTILABEL_CLASS[np.argmax(pred)])}")
		st.write(f"Confidence : {str(pred[0][np.argmax(pred)])}")
        
	else:
		best_acc = [-1,-1]
		st.image(test_image, caption="Input Image", width = 400)
		for i in range(len(DISEASE)):
			pred = predict_class(np.asarray(test_image), MODELS[i])
			st.write(f"Model {DISEASE[i]} : {str(pred)}")
			if best_acc[1]< pred[0][0]:
				best_acc=[i,pred[0][0]]
		st.divider()
		if best_acc[1]<.5:
			best_acc[0]='No Disease'
			st.write(f"Possible Result : {DISEASE[best_acc[0]]}")
	slot.text('Done')

