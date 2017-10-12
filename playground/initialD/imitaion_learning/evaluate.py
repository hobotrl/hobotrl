import numpy as np

def evaluate(y_true, y_preds, labels=[0, 1, 2, 3, 4]):
    """
    may there still has some problems. Need check more.
    :param y_true:
    :param y_preds:
    :param labels:
    :return:
    """
    p_scores = []
    r_scroes = []
    for label in labels:
        p = ((y_true == label) * (y_preds == label)).sum() / ((y_preds == label).sum() + 0.01)
        p_scores.append(p)
        r = ((y_true == label) * (y_preds == label)).sum() / ((y_true == label).sum() + 0.01)
        r_scroes.append(r)
    p_scores = np.array(p_scores)
    r_scroes = np.array(r_scroes)
    f1 = 2 * p_scores * r_scroes / (p_scores + r_scroes + 0.01)

    confmat = []
    for label in labels:
        conf = []
        for label2 in labels:
            conf.append(((y_preds == label) * (y_true == label2)).sum())
        confmat.append(conf)
    confmat = np.array(confmat)

    return p_scores, r_scroes, f1, confmat


def test_confmat():
    y_true = np.array([0,1,2,3,4,0,1,2,3,4])
    y_preds = np.array([0,2,1,3,2,0,1,2,1,0])
    p, r, f1, conf = evaluate(y_true, y_preds)
    print p
    print r
    print f1
    print conf


if __name__ == '__main__':
    test_confmat()