import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import minimize
from scipy.stats import spearmanr, kstest, probplot
import csv

# This project will be split into 3 sections, each with a different aim, ultimately coming together to form the whole
# Stage 1: Simulate a Hawkes Process using synthetic parameters and recover said parameters
# Stage 2: Estimate Hawkes Process parameters from historic AAPL message book data
# Stage 3: Validate parameters via KS test
# Stage 4: Compare and evaluate parameters for same and cross side excitation
# Data extracted from https://huggingface.co/datasets/totalorganfailure/lobster-data

# For stage 1, I will be synthetically simulating the Hawkes Process with synthetic parameters, and conditions, ultimately recovering these synthetic params

def simulate_thinning(alpha, beta, mu, t_max):
    """Since Hawkes has a decaying rate, we establish an upper-bound. However, this results in too large of a
    candidate pool, and so we must thin out this pool"""
    t = 0
    upper_int = mu
    history = []
    S = 0
    t_prev = 0
    while t < t_max:
        tau = np.random.exponential(1/upper_int)
        t_prev = t
        t += tau
        if t > t_max:
            break
        accept_prob = np.random.uniform(0, 1)
        S = S*np.exp(-beta*(t-t_prev))
        intensity_calc = mu + S
        if accept_prob <=  (intensity_calc/upper_int):
            S += alpha
            intensity_calc += alpha
            history.append(t)
            upper_int = intensity_calc
        else:
            upper_int = intensity_calc
    return np.array(history)

def intensity(alpha, beta, mu, history, t):
    """This computes the intensities of the Hawkes Process"""
    excite = 0
    for t_i in history:
        excite += alpha*np.exp(-(beta*(t - t_i)))
    return mu + excite

def set_ints(alpha, beta, mu, t_max):
    """A list of intensities along with their event times, and duration of the process is computed for more convenient
    plotting and recording of data"""
    duration = np.linspace(0, t_max, 50000)
    history = np.array(simulate_thinning(alpha, beta, mu, t_max))
    difference = duration[:, None] - history
    kernel = alpha*np.exp(-beta * difference)
    kernel = np.where(difference < 0, 0, kernel)
    excite = kernel.sum(axis = 1)
    list_ints = mu + excite
    return list_ints, history, duration

def synth_plot(alpha, beta, mu, t_max):
    fig, axs = plt.subplots(dpi = 300, figsize = (10, 6))
    plt.subplots_adjust(top=0.88)
    y, history, duration = set_ints(alpha, beta, mu, t_max)
    x = duration
    axs.plot(x, y, label = 'Conditional Intensity λ(t)')
    axs.plot(history, np.zeros_like(history), marker = '|', linestyle = 'none', label = 'Events', color = 'black')
    axs.set_ylabel('Intensity')
    axs.set_xlabel('Time')
    axs.set_title('Intensity as a Function of Time')
    fig.legend(loc = 'upper right', fontsize = '8')
    plt.show()

synth_plot(2, 4, 2, 10)

# This merely gives us a plot with pre-determined values of parameters
# An interactive, parameter changing plot is therefore necessary so that we can observe the changes to the plot as params change

def synth_plot_interact(alpha, beta, mu, t_max):
    fig, ax = plt.subplots(figsize = (10, 6), dpi = 300)
    fig.subplots_adjust(bottom=0.35, right=0.75)
    pos = ax.get_position()
    start_pos = pos.x0
    stop_pos = pos.width
    y, history, duration = set_ints(alpha, beta, mu, t_max)
    x = duration
    ax1 = fig.add_axes([start_pos, 0.2, stop_pos, 0.03])
    ax2 = fig.add_axes([start_pos, 0.15, stop_pos, 0.03])
    ax3 = fig.add_axes([start_pos, 0.1, stop_pos, 0.03])
    alpha_slide = Slider(ax1, 'alpha', valmin=0, valmax=10, valinit = alpha)
    beta_slide = Slider(ax2, 'beta', valmin=0, valmax=10, valinit = beta)
    mu_slide = Slider(ax3, 'mu', valmin=0, valmax=10, valinit = mu)
    def update_values(val):
        print(val)
        alpha_new = alpha_slide.val
        beta_new = beta_slide.val
        mu_new = mu_slide.val
        y_new, history_new, duration_new = set_ints(alpha_new, beta_new, mu_new, t_max)
        line.set_ydata(y_new)
        ticks.set_data(history_new, np.zeros_like(history_new))
        fig.canvas.draw_idle()
    alpha_slide.on_changed(update_values)
    beta_slide.on_changed(update_values)
    mu_slide.on_changed(update_values)
    line, = ax.plot(x, y, label = 'Conditional Intensity λ(t)')
    ticks, = ax.plot(history, np.zeros_like(history), marker = '|', linestyle = 'none', label = 'Events', color = 'black')
    ax.set_ylabel('Intensity')
    ax.set_xlabel('Time')
    ax.set_title('Intensity as a Function of Time')
    ax.legend(loc = 'upper left', bbox_to_anchor = (1.02, 1), fontsize = '7')
    plt.show()

# synth_plot_interact(2, 4, 2, 10)

# Now a function that can give the best fitting parameters is needed so that we can use it for the real data
# Best fitting, because the chance of the parameters being exact for a data set is negligibly low, but parameters can be recovered for the process in general

def parameter_score(params, event_times, t_max):
    """The parameter input should be a set of parameters in the order of alpha, beta, mu"""
    alpha, beta, mu = params
    S = 0
    t_prev = 0
    log_sum = 0
    integral = mu*t_max
    for t in event_times:
        S  = S*np.exp(-beta*(t-t_prev))
        log_sum += np.log(mu + S)
        S += alpha
        t_prev = t
        integral += (alpha/beta)*(1-np.exp(-beta*(t_max-t)))
    return log_sum - integral
# print(parameter_score([2, 4, 2], times, 10))

def best_params(params_0, event_times, t_max):
    """This allows best-fitting parameters to be returned, starting with an initial parameter guess"""
    def negative_score(params):
        params_exp = np.exp(params)
        score = parameter_score(params_exp, event_times, t_max)
        neg_score = -score
        return neg_score
    results = minimize(negative_score, np.log(params_0))
    return np.exp(results.x)

def time_sweep(times, alpha, beta, mu):
    """This allows a sweep through a list of t_max to be conducted, ultimately returning a list of event times"""
    datas = []
    for t_max in times:
        groups = []
        datas.append(groups)
        for i in range(5):
            results = simulate_thinning(alpha, beta, mu, t_max)
            groups.append(results)
    return datas

def params_sweep(times, alpha, beta, mu, params_0):
    """This then allows me to perform a sweep of the event time data set
    and return a list of bet fitting parameters for each t_max"""
    datas = time_sweep(times, alpha, beta, mu)
    param_list = []
    for t in range(len(times)):
        groups = []
        param_list.append(groups)
        for i in range(len(datas[t])):
            event_times = datas[t][i]
            t_max = times[t]
            parameters = best_params(params_0, event_times, t_max)
            groups.append(parameters)
    return param_list
# print(params_sweep([10, 100, 1000, 2000], 2, 4, 2, [2, 4, 2]))

def percentage_diff(times, alpha, beta, mu, params_0):
    param_list = np.array(params_sweep(times, alpha, beta, mu, params_0))
    params_true = np.array([alpha, beta, mu])
    p_diff = np.array((param_list - params_true)/(params_true) * 100)
    return p_diff

# print(percentage_diff([10, 100, 1000, 2000], 2, 4, 2, [2, 4, 2]))

def p_diff_plot(times, alpha, beta, mu, params_0):
    """Due to the large range in percentage difference observed in alpha and beta values,
    a symmetric log scaling has been used to create a more beneficial y-axis"""
    p_diff = percentage_diff(times, alpha, beta, mu, params_0)
    alpha_diff = p_diff[:, :, 0]
    beta_diff = p_diff[:, :, 1]
    mu_diff = p_diff[:, :, 2]
    times = np.repeat(times, 5)
    fig, axs = plt.subplots(3, 1, dpi = 300, figsize = (12, 8))
    axs[0].scatter(times, alpha_diff.flatten(), marker = '.')
    axs[0].set_xlabel('Maximum Times')
    axs[0].set_ylabel('Percentage Difference')
    axs[0].set_title('Alpha Percentage Differences at differing t_max values')
    axs[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axs[0].set_xscale('log')
    axs[0].set_yscale('symlog')
    axs[1].scatter(times, beta_diff.flatten(), marker = '.')
    axs[1].set_xlabel('Maximum Times')
    axs[1].set_ylabel('Percentage Difference')
    axs[1].set_title('Beta Percentage Differences at differing t_max values')
    axs[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axs[1].set_xscale('log')
    axs[1].set_yscale('symlog')
    axs[2].scatter(times, mu_diff.flatten(), marker = '.')
    axs[2].set_xlabel('Maximum Times')
    axs[2].set_ylabel('Percentage Difference')
    axs[2].set_title('Mu Percentage Differences at differing t_max values')
    axs[2].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axs[2].set_xscale('log')
    fig.suptitle('Parameter Percentage Differences at differing t_max values')
    plt.tight_layout()
    plt.show()

p_diff_plot([10, 100, 1000, 2000], 2, 4, 2, [2, 4, 2])

# So far, all functions have been univariate, for our AAPL message book, we are looking at bivariate data: buy and sell-side.
# I only really need time, event_type (specifically visible and hidden executions), and direction (buy or sell)

with open('AAPL_2012-06-21_34200000_37800000_message_50.csv') as apple_msg_50:
    reader = csv.reader(apple_msg_50, delimiter = ',')
    apple_data = []
    for row in reader:
        if int(row[1]) == 4 or int(row[1]) == 5:
            apple_data.append(row)
    AAPL_data = np.array(apple_data)
    filtered_data = AAPL_data[:,[0,1,3,5]]
# print(filtered_data)

def group_orders(data):
    """This function aims to aggregate simultaneous, same-direction executions which indicate one aggressive order.
    They are classified by the executions with the same direction made within the same time"""
    time_size_direction = data[:,[0,2,3]]
    classification = {}
    for row in time_size_direction:
        key = tuple(row[[0,2]])
        classification.setdefault(key,[0, 0])
        classification[key][0] += 1
        classification[key][1] += int(row[1])
    return classification
# print(group_orders(filtered_data))

def conversion(data):
    """Since the group_orders() function returns a dictionary and not an array with usable numbers for data analysis,
    we must convert this dictionary into an array."""
    dictionary = group_orders(data)
    final_data = []
    for entry in dictionary.items():
        conversion_time = float(entry[0][0])
        conversion_direction = int(entry[0][1])
        count = entry[1][0]
        volume = entry[1][1]
        final_data.append([conversion_time,conversion_direction, count, volume])
    return sorted(final_data)
# print(conversion(filtered_data))
final_data = conversion(filtered_data)

def ratio(data):
    """This function shows the ratio between the length of data after and before grouping the executions
    made by the same entity at the same time"""
    final_data = conversion(data)
    comparisons = len(final_data)/len(filtered_data)
    return comparisons
# print(ratio(filtered_data))
# Note that the ratio was 0.723 correct to 3 d.p
# This means around 27.7% of events were absorbed into singular events

def data_grouping(data):
    """In order to make stage 2 easier, grouping the data into buy and sell-side is crucial
    which is what this function does, grouping the data by direction: 1 = sell, -1 = buy. It returns a list of two lists:
    sell-side and buy-side"""
    buy_side = []
    sell_side = []
    for rows in data:
        if rows[1] == 1:
            sell_side.append(rows)
        else:
            buy_side.append(rows)
    return sell_side, buy_side
# print(data_grouping(final_data))

# Since the parameter_score and best_params functions require event times, a function should be created to extract and group only the timings of executions.

def event_times_group(data):
    sell_time = []
    buy_time = []
    sell_side, buy_side = data_grouping(data)
    for rows in sell_side:
        sell_time.append(rows[0])
    for rows in buy_side:
        buy_time.append(rows[0])
    return sell_time, buy_time
# print(event_times_group(final_data))

# Now we create modifications to our univariate parameter recoverer/estimators to transform them into bivariate functions
sell_time, buy_time = event_times_group(filtered_data)

def biv_param_score(params, data, t_max):
    """The parameters now are VERY different. There are now four distinct alpha values, and two distinct mu values.
    The elements in the params therefore must follow as such:
    alpha values of sell-sell, buy-sell, buy-buy, sell-buy; beta value, mu values of sell and buy.
    It may seem counterintuitive to update some buy values in the sell branch and vice versa, but this is because an action
    in said branch influences the result of another and must be included in the causal branch"""
    alpha_ss, alpha_bs, alpha_bb, alpha_sb, beta, mu_s, mu_b = params
    sum_sell = 0
    sum_buy = 0
    log_sum_sell = 0
    log_sum_buy = 0
    integral_sell = mu_s*t_max
    integral_buy = mu_b*t_max
    t_prev = 0
    for rows in data:
        t = rows[0]-34200
        direction = rows[1]
        sum_sell = sum_sell*np.exp(-beta*(t-t_prev))
        sum_buy = sum_buy*np.exp(-beta*(t-t_prev))
        t_prev = t
        sell_int = mu_s + sum_sell
        buy_int = mu_b + sum_buy
        if direction == 1:
            log_sum_sell += np.log(sell_int)
            sum_sell += alpha_ss
            sum_buy += alpha_sb
            integral_sell += (alpha_ss/beta)*(1-np.exp(-beta*(t_max-t)))
            integral_buy += (alpha_sb/beta)*(1-np.exp(-beta*(t_max-t)))
        else:
            log_sum_buy += np.log(buy_int)
            sum_buy += alpha_bb
            sum_sell += alpha_bs
            integral_buy += (alpha_bb / beta) * (1 - np.exp(-beta * (t_max - t)))
            integral_sell += (alpha_bs / beta) * (1 - np.exp(-beta * (t_max - t)))
        log_lh_sell = log_sum_sell - integral_sell
        log_lh_buy = log_sum_buy - integral_buy
        total_log_lh = log_lh_sell + log_lh_buy
    return total_log_lh

def biv_best_params(params_0, data, t_max):
    def negative_score(params):
        params_exp = np.exp(params)
        score = biv_param_score(params_exp, data, t_max)
        neg_score = -score
        return neg_score
    results = minimize(negative_score, np.log(params_0))
    return np.exp(results.x)

# Before that, we can not blindly guess the initial parameters we must put it. We can use this equation to 'estimate' them: μ = Λ(1-n)
# Let us assume n = 0.5 as a middle ground, and we can work out Λ manually for each direction

sell_side, buy_side = data_grouping(final_data)
sell_obs_rate = len(sell_side)/3600
buy_obs_rate = len(buy_side)/3600
# print(sell_obs_rate, buy_obs_rate)

mu_sell = sell_obs_rate*0.5
mu_buy = buy_obs_rate*0.5
# print(mu_sell, mu_buy)

AAPL_params = biv_best_params([2.5, 2.5, 2.5, 2.5, 5, mu_sell, mu_buy], final_data, 3600)
alpha_ss, alpha_bs, alpha_bb, alpha_sb, beta_apple, m_sell, m_buy = AAPL_params
n_ss = alpha_ss/beta_apple
n_bb = alpha_bb/beta_apple
n_bs = alpha_bs/beta_apple
n_sb = alpha_sb/beta_apple
# print(n_ss, n_bb, n_bs, n_sb, m_sell, m_buy)

# We can compare simulated Hawkes data, parameters recovered from the simualted data, and the actual estimated parameters
# This means we must also augment simulate_thinning and intensity into a bivariate format

def biv_simulate_thinning(params, t_max):
    a_ss, a_bs, a_bb, a_sb, b, m_s, m_b = params
    t = 0
    upper_int = m_s + m_b
    history_sell = []
    history_buy = []
    sum_s = 0
    sum_b = 0
    t_prev = 0
    while t < t_max:
        tau = np.random.exponential(1/upper_int)
        t_prev = t
        t += tau
        if t > t_max:
            break
        sum_s = sum_s*np.exp(-b*(t-t_prev))
        sum_b = sum_b * np.exp(-b*(t-t_prev))
        intensity_calc_s = m_s + sum_s
        intensity_calc_b = m_b + sum_b
        pooled_int = intensity_calc_s + intensity_calc_b
        accept_prob = np.random.uniform(0,1)
        if accept_prob <=  ((intensity_calc_s+intensity_calc_b)/upper_int):
            sell_prob = np.random.uniform(0, 1)
            if sell_prob <= intensity_calc_s/(pooled_int):
                sum_s += a_ss
                sum_b += a_sb
                intensity_calc_s += a_ss
                intensity_calc_b += a_sb
                history_sell.append(t)
                upper_int = intensity_calc_s + intensity_calc_b
            else:
                sum_b += a_bb
                sum_s += a_bs
                intensity_calc_b += a_bb
                intensity_calc_s += a_bs
                history_buy.append(t)
                upper_int = intensity_calc_b + intensity_calc_s
        else:
            upper_int = pooled_int
    dir_time_sell = [[t, 1] for t in history_sell]
    dir_time_buy = [[t, -1] for t in history_buy]
    comb_times = dir_time_sell + dir_time_buy
    comb_data = np.array(comb_times)
    order = np.argsort(comb_data[:, 0])
    comb_data = comb_data[order]
    return comb_data, dir_time_sell, dir_time_buy, np.array(history_sell), np.array(history_buy)

def biv_set_ints(params, t_max):
    a_ss, a_bs, a_bb, a_sb, b, m_s, m_b = params
    duration = np.linspace(0, t_max, 50000)
    comb_data, dir_time_sell, dir_time_buy, history_sell, history_buy = biv_simulate_thinning(params, t_max)
    difference_sell = duration[:, None] - history_sell
    difference_buy = duration[:, None] - history_buy
    difference_sell = np.where(difference_sell < 0, np.inf, difference_sell)
    difference_buy = np.where(difference_buy < 0, np.inf, difference_buy)
    kernel_ss = a_ss * np.exp(-b * difference_sell)
    kernel_sb = a_sb * np.exp(-b * difference_sell)
    kernel_bb = a_bb * np.exp(-b * difference_buy)
    kernel_bs = a_bs * np.exp(-b * difference_buy)
    kernel_ss = np.where(difference_sell < 0, 0, kernel_ss)
    kernel_sb = np.where(difference_sell < 0, 0, kernel_sb)
    kernel_bb = np.where(difference_buy < 0, 0, kernel_bb)
    kernel_bs = np.where(difference_buy < 0, 0, kernel_bs)
    excite_sell = kernel_ss.sum(axis=1) + kernel_bs.sum(axis=1)
    list_ints_sell = m_s + excite_sell
    excite_buy = kernel_bb.sum(axis=1) + kernel_sb.sum(axis=1)
    list_ints_buy = m_b + excite_buy
    return list_ints_sell, list_ints_buy, history_sell, history_buy, duration

def apple_plot(params, t_max):
    y1, y2, h_s, h_b, x= biv_set_ints(params, t_max)
    fig, axs = plt.subplots(2,1, dpi = 300, figsize = (12, 8))
    plt.subplots_adjust(top=0.88)
    line_int, = axs[0].plot(x, y1, label = 'Conditional Sell Intensity λ(t)')
    events, = axs[0].plot(h_s, np.zeros_like(h_s), marker='|', markersize = 5, linestyle='none', label='Events', color='black')
    axs[0].set_title('Sell Stream')
    axs[1].plot(x, y2, label = 'Conditional Buy Intensity λ(t)')
    axs[1].plot(h_b, np.zeros_like(h_b), marker='|', markersize = 5, linestyle='none', label='Events', color='black')
    axs[1].set_title('Buy Stream')
    fig.supylabel('Intensity')
    fig.supxlabel('Time')
    fig.suptitle('Intensity as a Function of Time')
    fig.legend([line_int, events], ['Conditional Intensity λ(t)', 'Execution Volume'], loc='upper right')
    plt.show()

apple_plot(AAPL_params,  3600)

# Now that we have estimated parameters recovered, we should validate them with the KS test

def time_rescale(params, data):
    a_ss, a_bs, a_bb, a_sb, b, m_s, m_b = params
    t_prev = 0
    sell_stream = []
    buy_stream = []
    sell_const = 0
    sell_decay = 0
    buy_const = 0
    buy_decay = 0
    for rows in data:
        t = rows[0]-34200
        direction = rows[1]
        sell_decay = sell_decay * np.exp(-b * (t - t_prev))
        buy_decay = buy_decay * np.exp(-b * (t - t_prev))
        t_prev = t
        if direction == 1:
            sell_terms = m_s * t + (sell_const - sell_decay)
            sell_stream.append(sell_terms)
            sell_const += a_ss / b
            sell_decay += a_ss / b
            buy_const += a_sb / b
            buy_decay += a_sb / b
        else:
            buy_terms = m_b * t + (buy_const - buy_decay)
            buy_stream.append(buy_terms)
            buy_const += a_bb / b
            buy_decay += a_bb / b
            sell_const += a_bs / b
            sell_decay += a_bs / b
    return np.array(sell_stream), np.array(buy_stream)

def term_difference(params, data):
    sell_stream, buy_stream = time_rescale(params, data)
    sell_gaps = np.diff(sell_stream)
    buy_gaps = np.diff(buy_stream)
    return sell_gaps, buy_gaps

def params_validate(params, data):
    sell_gaps, buy_gaps = term_difference(params, data)
    sell_params_confirm = kstest(sell_gaps, 'expon', args = (0, 1.0))
    buy_params_confirm = kstest(buy_gaps, 'expon', args = (0, 1.0))
    return sell_params_confirm, buy_params_confirm

print(params_validate(AAPL_params, final_data))
# the p_values of both sell and buy streams showed that the exponential kernel was a bad fit i.e. negligible values

def gaps_mean(params, data):
    sell_gaps, buy_gaps = term_difference(params, data)
    sell_means = np.mean(sell_gaps)
    buy_means = np.mean(buy_gaps)
    return sell_means, buy_means
# print(gaps_mean(AAPL_params, final_data)
# Note that the mean of the gaps were close to 1, meaning the scaling has no problem

def gaps_plot(params, data):
    sell_gaps, buy_gaps = term_difference(params, data)
    fig, axs = plt.subplots(2, 1, dpi = 300, figsize = (12, 8))
    probplot(sell_gaps, dist = 'expon', plot = axs[0])
    axs[0].lines[0].set_marker('.')
    axs[0].lines[0].set_label('Observed Quantiles')
    axs[0].lines[1].set_color('orange')
    axs[0].lines[1].set_label('Fitted line')
    axs[0].axline((0, 0), slope=1, color='black', linestyle='--', label = 'Theoretical Reference (y=x)')
    axs[0].set_title('Sell Stream Probability Plot')
    axs[0].legend(loc = 'upper left')
    axs[0].set_xlim(left = 0)
    axs[0].set_ylim(bottom = 0)
    probplot(buy_gaps, dist = 'expon', plot = axs[1])
    axs[1].lines[0].set_marker('.')
    axs[1].lines[0].set_label('Observed Quantiles')
    axs[1].lines[1].set_color('orange')
    axs[1].lines[1].set_label('Fitted line')
    axs[1].axline((0, 0), slope=1, color='black', linestyle='--', label = 'Theoretical Reference (y=x)')
    axs[1].set_title('Buy Stream Probability Plot')
    axs[1].legend(loc = 'upper left')
    axs[1].set_xlim(left = 0)
    axs[1].set_ylim(bottom = 0)
    fig.tight_layout()
    plt.show()

gaps_plot(AAPL_params, final_data)

def branching_rat(params):
    a_ss, a_bs, a_bb, a_sb, b, m_s, m_b = params
    branch_ss = a_ss/b
    branch_bs = a_bs/b
    branch_bb = a_bb/b
    branch_sb = a_sb/b
    return branch_ss, branch_bs, branch_bb, branch_sb

print(branching_rat(AAPL_params))

def bar_branching_rat(params):
    ss, bs, bb, sb = branching_rat(params)
    plt.figure(dpi=300, figsize=(10, 6))
    plt.bar(['sell-sell', 'buy-buy', 'sell-buy', 'buy-sell'], [ss, bb, sb, bs])
    plt.xlabel('Execution Type')
    plt.ylabel('Branching Ratio')
    plt.title('Branching Ratio of each Execution Type')
    plt.show()

bar_branching_rat(AAPL_params)

# An interesting extension is evaluating the correlation between execution volume and execution intensity
# We will first evaluate simultaneous execution volume and intensity, then lagging volume and current intensity
# The correlation coefficient we will use is the Spearman Rank Correlation Coefficient
def execution_volume(data):
    sell_side, buy_side = data_grouping(data)
    sell_execution = []
    buy_execution = []
    for rows in sell_side:
        sell_execution.append(rows[3])
    for rows in buy_side:
        buy_execution.append(rows[3])
    return sell_execution, buy_execution

def event_times_ints(params, data):
    a_ss, a_bs, a_bb, a_sb, b, m_s, m_b = params
    sum_sell = 0
    sum_buy = 0
    t_prev = 0
    sell_stream = []
    buy_stream = []
    for rows in data:
        t = rows[0]-34200
        direction = rows[1]
        sum_sell = sum_sell * np.exp(-b * (t - t_prev))
        sum_buy = sum_buy * np.exp(-b * (t - t_prev))
        t_prev = t
        sell_int = m_s + sum_sell
        buy_int = m_b + sum_buy
        if direction == 1:
            sell_stream.append([t, sell_int])
            sum_sell += a_ss
            sum_buy += a_sb
        else:
            buy_stream.append([t, buy_int])
            sum_buy += a_bb
            sum_sell += a_bs
    return np.array(sell_stream), np.array(buy_stream)

def vol_int_data_prep(params, data):
    sell_execution, buy_execution = execution_volume(data)
    sell_stream, buy_stream = event_times_ints(params, data)
    t_sell = sell_stream[:, 0]
    t_buy = buy_stream[:, 0]
    sell_int = sell_stream[:, 1]
    buy_int = buy_stream[:, 1]
    return sell_execution, buy_execution, sell_int, buy_int, t_sell, t_buy

def int_volume_plot(params, data):
    sell_execution, buy_execution, sell_int, buy_int, t_sell, t_buy = vol_int_data_prep(params, data)
    fig, axs = plt.subplots(2, 1, dpi=300, figsize=(12, 8))
    plt.subplots_adjust(top=0.88)
    axs_num_s = axs[0].twinx()
    axs_num_b = axs[1].twinx()
    line_int, = axs[0].plot(t_sell, sell_int, label = 'Conditional Sell Intensity λ(t)', color = 'blue', alpha = 0.5)
    line_vol, = axs_num_s.plot(t_sell, sell_execution, label = 'Volume of Sell Executions', color = 'orange', marker = '.', linestyle = 'none', markersize = 3)
    axs[0].set_title('Conditional Sell Intensity λ(t) & Sell Execution Volume against Time')
    axs[1].plot(t_buy, buy_int, label = 'Conditional Buy Intensity λ(t)', color = 'blue', alpha = 0.5)
    axs_num_b.plot(t_buy, buy_execution, label = 'Volume of Buy Executions', color = 'orange', marker = '.', linestyle = 'none', markersize = 3)
    axs[1].set_title('Conditional Buy Intensity λ(t) & Buy Execution Volume against Time')
    axs[0].set_ylabel('Intensity')
    axs_num_s.set_ylabel('Execution Volume')
    axs[1].set_ylabel('Intensity')
    axs_num_b.set_ylabel('Execution Volume')
    fig.supxlabel('Time')
    fig.legend([line_int, line_vol], ['Conditional Intensity λ(t)', 'Execution Volume'], loc = 'upper right')
    fig.suptitle('Contemporaneous Volume and Intensity plot')
    plt.show()

int_volume_plot([alpha_ss, alpha_bs, alpha_bb, alpha_sb, beta_apple, m_sell, m_buy], final_data)

def spearman_vol_int(params, data):
    sell_vol, buy_vol = execution_volume(data)
    sell_stream, buy_stream = event_times_ints(params,data)
    sell_int = sell_stream[:, 1]
    buy_int = buy_stream[:, 1]
    sell_srcc = spearmanr(sell_vol, sell_int)
    buy_srcc = spearmanr(buy_vol, buy_int)
    return sell_srcc, buy_srcc

print(spearman_vol_int(AAPL_params, final_data))

def vol_int_slicing_data(params, data):
    """A preparation of lagged execution volumes to current event intensities"""
    sell_stream, buy_stream = event_times_ints(params, data)
    sell_execution, buy_execution = execution_volume(data)
    t_sell = sell_stream[:, 0]
    t_buy = buy_stream[:, 0]
    vol_t_sell = t_sell[:-1]
    vol_t_buy = t_buy[:-1]
    int_t_sell = t_sell[1:]
    int_t_buy = t_buy[1:]
    sell_int = sell_stream[1:, 1]
    buy_int = buy_stream[1:, 1]
    sell_vol = sell_execution[:-1]
    buy_vol = buy_execution[:-1]
    return sell_vol, buy_vol, sell_int, buy_int, vol_t_sell, vol_t_buy, int_t_sell, int_t_buy

def lagging_vol_plot(params, data):
    sell_vol, buy_vol, sell_int, buy_int, vol_t_sell, vol_t_buy, int_t_sell, int_t_buy = vol_int_slicing_data(params, data)
    fig, axs = plt.subplots(2, 1, dpi=300, figsize=(12, 8))
    plt.subplots_adjust(top=0.88)
    axs_num_s = axs[0].twinx()
    axs_num_b = axs[1].twinx()
    line_int, = axs[0].plot(int_t_sell, sell_int, label='Conditional Sell Intensity λ(t)', color='blue', alpha=0.5)
    line_vol, = axs_num_s.plot(vol_t_sell, sell_vol, label='Volume of Sell Executions', color='orange', marker='.', linestyle='none', markersize=3)
    axs[0].set_title('Conditional Sell Intensity λ(t) & Sell Execution Volume against Time')
    axs[1].plot(int_t_buy, buy_int, label='Conditional Buy Intensity λ(t)', color='blue', alpha=0.5)
    axs_num_b.plot(vol_t_buy, buy_vol, label='Volume of Buy Executions', color='orange', marker='.', linestyle='none', markersize=3)
    axs[1].set_title('Conditional Buy Intensity λ(t) & Buy Execution Volume against Time')
    axs[0].set_ylabel('Intensity')
    axs_num_s.set_ylabel('Execution Volume')
    axs[1].set_ylabel('Intensity')
    axs_num_b.set_ylabel('Execution Volume')
    fig.supxlabel('Time')
    fig.legend([line_int, line_vol], ['Conditional Intensity λ(t)', 'Execution Volume'], loc='upper right')
    fig.suptitle('Lagged Volume and Intensity plot')
    plt.show()

lagging_vol_plot(AAPL_params, final_data)

def spearman_lagging_vol_int(params, data):
    sell_vol, buy_vol, sell_int, buy_int, vol_t_sell, vol_t_buy, int_t_sell, int_t_buy = vol_int_slicing_data(params, data)
    sell_srcc_lag = spearmanr(sell_vol, sell_int)
    buy_srcc_lag = spearmanr(buy_vol, buy_int)
    return sell_srcc_lag, buy_srcc_lag

print(spearman_lagging_vol_int(AAPL_params, final_data))


