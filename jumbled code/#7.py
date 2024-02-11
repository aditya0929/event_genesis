#7

x_imgs = []
y_lbls = []

for sfldr in tqdm(os.listdir('/kaggle/input/asl-dataset/asl_dataset')):
    sfldr_path = os.path.join('/kaggle/input/asl-dataset/asl_dataset', sfldr)
    if not os.path.isdir(sfldr_path) or sfldr == 'asl_dataset':
        continue
    for img_filename in os.listdir(sfldr_path):
        img_path = os.path.join(sfldr_path, img_filename)
        x_imgs.append(img_path)
        y_lbls.append(sfldr)

df_sign_language = pd.DataFrame({'image': x_imgs, 'label': y_lbls})

plt.figure(figsize=(15, 6))
for n, i in enumerate(np.random.randint(0, len(df_sign_language), 10)):
    plt.subplot(2, 5, n + 1)
    img = cv2.imread(df_sign_language.image[i])
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.title(df_sign_language.label[i], fontsize=10)

plt.show()
