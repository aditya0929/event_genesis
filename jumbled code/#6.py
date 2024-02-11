#6
checkpoint_cb = ModelCheckpoint("sign_lang_model.h5", save_best_only=True)
early_stopping_cb = EarlyStopping(patience=5, restore_best_weights=True)
sign_lang_model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = sign_lang_model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=[checkpoint_cb, early_stopping_cb])