from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def label_classification(
    embeds, test_embeds, train_g, test_g
):
    X = embeds.clone().detach().cpu().numpy()
    y = train_g.ndata['labels']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        random_state=1)

    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    clf.predict_proba(X_test[:1])

    clf.predict(X_test[:5, :])

    y_pred = clf.predict_proba(test_embeds.cpu())

    acc = (y_pred.argmax(-1) == test_g.ndata["labels"].detach().numpy()).mean()
    return clf, acc

    