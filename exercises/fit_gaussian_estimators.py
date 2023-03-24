import numpy as np
import numpy.random
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian

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
                       title="Distance Between Estimated And True "
                             "Value Of The Expectation",
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
                       title="PDF Of The Previous Drawn Samples",
                       )
    fig_3.write_image("q3.result.png")


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = [0, 0, 4, 0]
    cov = np.array(
        [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    samples = numpy.random.multivariate_normal(mu, cov, 1000)
    mg = MultivariateGaussian()
    mg.fit(samples)
    print(mg.mu_)
    print(mg.cov_)

    # # Question 5 - Likelihood evaluation
    linspace = np.linspace(-10, 10, 200)
    log_likelihood_result = np.zeros((200, 200))
    for i in range(len(linspace)):
        for j in range(len(linspace)):
            mu = np.array([linspace[i], 0, linspace[j], 0]).T
            likelihood_result = MultivariateGaussian. \
                log_likelihood(mu, cov, samples)
            log_likelihood_result[i][j] = likelihood_result

    fig_5 = go.Figure(
        [go.Heatmap(x=linspace, y=linspace, z=log_likelihood_result, )],
    )
    fig_5.update_layout(
        title='Log-Likelihood Where Mu Is [f1,0,f3,0]',
        xaxis_title="$\\text{f3}$",
        yaxis_title="$\\text{f1}$"
    )
    fig_5.write_image("q5.result.png")

    # # Question 6 - Maximum likelihood
    max_index = np.argmax(log_likelihood_result)
    max_index_row = int(np.floor(max_index / 200))
    max_index_col = (max_index - max_index_row * 200) % 200
    print("f1: " + str(round(linspace[max_index_row], 3)))
    print("f3: " + str(round(linspace[max_index_col], 3)))
    print(log_likelihood_result[max_index_row][max_index_col])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
