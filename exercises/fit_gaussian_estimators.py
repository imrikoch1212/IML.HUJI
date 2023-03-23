import numpy as np
import plotly.express as px
import plotly.io as pio

from IMLearn.learners import UnivariateGaussian

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    ug = UnivariateGaussian()
    mu = 10
    var = 1
    x = np.random.normal(mu, var, 1000)
    ug.fit(x)
    empirical = UnivariateGaussian().fit(x)
    print((ug.mu_, ug.var_))

    # # Question 2 - Empirically showing sample mean is consistent
    question_2_results = []
    for i in range(1, int(len(x) / 10)):
        question_2_results.append(np.abs(ug.fit(x[:i * 10]).mu_ - mu))
    fig_2 = px.scatter(x=[list(range(10, 1000, 10))], y=question_2_results,
                       labels={
                           "x": "Sample Size",
                           "y": "Distance from mu"
                       },
                       title="Sample mean estimation deviation",
                       )
    fig_2.write_image("q2.result.png")

    # # Question 3 - Plotting Empirical PDF of fitted model

    pdfs = empirical.pdf(x)
    # graph_data = np.c_[x, pdfs]
    fig_3 = px.scatter(x=x, y=pdfs,
                       labels={
                           "x": "x",
                           "y": "PDF"
                       },
                       title="Empirical PDF",
                       )
    fig_3.write_image("q3.result.png")


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()
