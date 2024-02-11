#3
score, acc = sign_lang_model.evaluate(test_generator)
print('Test Loss =', score)
print('Test Accuracy =', acc)
y_test = test_generator.classes
predictions = sign_lang_model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_test = np.ravel(y_test)
y_pred = np.ravel(y_pred)
df_result = pd.DataFrame({'Actual': y_test, 'Prediction': y_pred})
classification_report_result = classification_report(y_test, y_pred)
print('Classification Report is : ', classification_report_result)