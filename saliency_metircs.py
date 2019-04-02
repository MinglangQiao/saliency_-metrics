def calc_cc_score(gtsAnn, resAnn):
    """
    Computer CC score. A simple implementation
    :param gtsAnn : ground-truth fixation map
    :param resAnn : predicted saliency map
    :return score: int : score
    """

    fixationMap = gtsAnn - np.mean(gtsAnn)

    if np.max(fixationMap) > 0:
        fixationMap = fixationMap / np.std(fixationMap)
    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)

    return np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]

def calc_kl_score(gtsAnn, resAnn,eps = 1e-7):
    if np.sum(gtsAnn) > 0:
        gtsAnn = gtsAnn / np.sum(gtsAnn)
    if np.sum(resAnn) > 0:
        resAnn = resAnn / np.sum(resAnn)
    return np.sum(gtsAnn * np.log(eps + gtsAnn / (resAnn + eps)))
