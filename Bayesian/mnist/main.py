from dataloader import load_train_images,load_train_labels,load_test_images,load_test_labels
from featureExtraction import featureExtraction
from train import bayesModelTrain
from evaluation import modelEvaluation

if __name__ == '__main__':
    print('loading MNIST Data')
    train_images = load_train_images()

    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()
    print('loading done')

    nowMnistLabel = train_labels[0].copy()
    nowMnistImage = train_images[0].copy()
    print(nowMnistLabel)
    # plt.imshow(nowMnistImage, cmap='gray')
    # plt.pause(0.001)
    # plt.show()

    print('feature extraction start')
    train_images_feature = featureExtraction(train_images)
    print('feature extraction done')
    nowMnistLabel = train_labels[0].copy()
    nowMnistImage = train_images_feature[0].copy()
    print(nowMnistLabel)
    # plt.imshow(nowMnistImage, cmap='gray')
    # plt.pause(0.001)
    # plt.show()

    print('bayes model train start')
    prioriP, posteriorP = bayesModelTrain(train_images_feature, train_labels)
    print('bayes model train done')
    # print(prioriP)
    # print(posteriorP)

    print('bayes model evaluation start')
    test_images_feature = featureExtraction(test_images)
    res, val = modelEvaluation(
        test_images_feature, test_labels, prioriP, posteriorP)
    print('贝叶斯分类器的准确度为%.2f %%' % (val*100))
    print('bayes model evaluation done')