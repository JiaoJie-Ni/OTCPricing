import numpy as np

TRADING_DAY = 252
r = 0.03


def monte_carlo(S0, S, sigma, dt, T, knock_in, knock_out, coupon, n):
    def cal_settle_price(row):
        tmp_up_d = np.where(row > S0 * knock_out)[0]
        tmp_up_m = tmp_up_d[(tmp_up_d + 1) % 21 == 0]
        tmp_dn_d = np.where(row < S0 * knock_in)[0]

        if len(tmp_up_m) > 0:
            t = (tmp_up_m[0] + 1) / TRADING_DAY
            payoff = coupon * t * np.exp(-r * t)
        elif len(tmp_dn_d) > 0:
            payoff = np.minimum(row[-1] - S0, 0) * np.exp(-r * T)
        else:
            payoff = coupon * np.exp(-r * T)
        return payoff
    
    def cal_premium(randnum):
        path = S * np.cumprod(np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * randnum), axis=1)
        average = np.apply_along_axis(cal_settle_price, axis=1, arr=path)
        return average.mean() / np.exp(-r * T / TRADING_DAY)
    
    np.random.seed(0)
    random_number = np.random.normal(0, 1, [n, int(T / dt)])
    return 0.5 * (cal_premium(random_number) + cal_premium(-random_number))


# def delta(S0, S, r, sigma, t, T, K):
#     delta = (monte_carlo(S0, S * (1 + sigma), 0, sigma, 1, T, K, 300000) -
#              monte_carlo(S0, S * (1 - sigma), 0, sigma, 1, T, K, 300000)) / (S * sigma * 2)
#     return delta
#
#
# def gamma_sr(S0, S, r, sigma, t, T, K):
#     gamma = (delta(S0, S * (1 + sigma), r, sigma, t, T, K) -
#              delta(S0, S * (1 - sigma), r, sigma, t, T, K)) / (S * sigma * 2)
#     return gamma
#
#
# def vega_sr(S0, S, r, sigma, t, T, K):
#     vega = (monte_carlo(S0, S, 0, sigma + 0.0001, 1, T, K, 300000) -
#             monte_carlo(S0, S, 0, sigma - 0.0001, 1, T, K, 300000)) / 2
#     return vega


if __name__ == '__main__':
    sigma_sr = 0.13
    S0_sr = 1
    S_sr = 1
    T = 1
    path_num = 300000

    premium_sr_today = monte_carlo(S0_sr, S_sr, sigma_sr, 1 / TRADING_DAY, T, 0.85, 1.03, 0.2, path_num)
    print("核算报价_sr=", premium_sr_today)
