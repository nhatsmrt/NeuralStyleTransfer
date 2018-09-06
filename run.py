def run(num_epoch = 1, batch_size=16, print_every=200, IMG_SIZE = 256, N_IMG_MAX = 20000, use_previous_weights = False):
    from Source import StyleTransferNet
    from sklearn.model_selection import train_test_split
    import numpy as np
    import os
    import cv2
    from pathlib import Path


    # DEFINE PATHS:
    d = Path().resolve()
    data_path = str(d) + "/"
    predictions_path = str(d) + "/Predictions/"
    style_path = data_path + "Mouse.png"
    train_path = data_path + "MiniCOCO/" + str(IMG_SIZE) + "/"
    weight_save_path = str(d) + "/weights/model.ckpt"
    weight_load_path = str(d) + "/weights/model.ckpt"
    pretrained_path = str(d) + "/imagenet-vgg-verydeep-19.mat"

    style_img = cv2.imread(style_path).astype(np.float32)

    X = []
    names = []
    model = StyleTransferNet(
        style_img,
        pretrained_path=pretrained_path
    )

    n_img_processed = 0
    for filename in os.listdir(train_path):
        if not (filename.endswith(".jpg") or filename.endswith(".png")):
            continue

        img = cv2.imread(train_path + filename)
        names.append(filename)
        X.append(img)
        n_img_processed += 1

        if n_img_processed == N_IMG_MAX:
            break

    X = np.array(X)
    names = np.array(names)
    print(X.shape[0])
    X_train_val, X_test, names_train_val, names_test = train_test_split(X, names, test_size=0.20)
    X_train, X_val = train_test_split(X_train_val, test_size = 0.10)

    # mean = np.mean(X_train, axis = 0)
    # std = np.mean(X_train, axis = 0)
    #
    # X_train = (X_train - mean) / std
    # X_test = (X_test - mean) / std

    if use_previous_weights:
        model.load_weights(weight_load_path)

    model.fit(
        X_train,
        X_val = X_val,
        weight_save_path = weight_save_path,
        num_epoch = num_epoch,
        batch_size = batch_size,
        print_every = print_every
    )
    model.load_weights(weight_load_path)
    predictions = model.predict(X_test)

    for ind in range(X_test.shape[0]):
        filename = predictions_path + names_test[ind][:-4] + "_styled.png"
        original_file = predictions_path + names_test[ind]

        cv2.imwrite(filename, predictions[ind])
        cv2.imwrite(original_file, X_test[ind])

run(num_epoch = 100)