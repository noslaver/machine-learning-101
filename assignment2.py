#################################
# Your name:
#################################

import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        points = np.random.rand(m, 2)
        for i in range(m):
            x, _ = points[i]
            if 0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1:
                points[i][1] = bernoulli.rvs(0.8)
            else:
                points[i][1] = bernoulli.rvs(0.1)

        return points

    def draw_sample_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: None.
        """
        points = self.sample_from_D(m)
        points = points[points[:,0].argsort()] # sort by x
        xs = points[:,0]
        ys = points[:,1]

        best_intervals, error = intervals.find_best_interval(xs, ys, k)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis([0, 1, -0.1, 1.1])

        plt.axvline(x=0.2, linestyle='--', color='k')
        plt.axvline(x=0.4, linestyle='--', color='k')
        plt.axvline(x=0.6, linestyle='--', color='k')
        plt.axvline(x=0.8, linestyle='--', color='k')

        zeros = [point[0] for point in points if point[1] == 0]
        ones = [point[0] for point in points if point[1] == 1]
        plt.plot(zeros, np.zeros(len(zeros)), 'x', label = '0')
        plt.plot(ones, np.ones(len(ones)), 'o', label = '1')

        for interval in best_intervals:
            plt.hlines(0, interval[0], interval[1], '#ccffff', lw=1000)

        plt.savefig('a.png')

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        errors = np.ndarray((math.floor((m_last - m_first) / step), 2))
        ms = range(m_first, m_last, step)
        for i, m in enumerate(ms):
            empirical_error = 0
            true_error = 0
            for _ in range(T):
                points = self.sample_from_D(m)
                points = points[points[:,0].argsort()] # sort by x
                xs = points[:,0]
                ys = points[:,1]

                best_intervals, test_errors = intervals.find_best_interval(xs, ys, k)

                empirical_error += self.calculate_empirical_error(m, test_errors)
                true_error += self.calculate_true_error(best_intervals)

            empirical_error /= T
            true_error /= T
            errors[i] = [empirical_error, true_error]

        plt.xlabel('m')
        plt.ylabel('error')

        empirical_errs = errors[:,0]
        true_errs = errors[:,1]
        plt.plot(ms, empirical_errs, 'x', label = 'empirical error')
        plt.plot(ms, true_errs, 'o', label = 'true error')
        plt.legend(loc='upper right')

        plt.savefig('c.png')
        return errors

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        errors = np.ndarray((math.floor((k_last + 1 - k_first) / step), 2))
        ks = range(k_first, k_last + 1, step)
        points = self.sample_from_D(m)
        points = points[points[:,0].argsort()] # sort by x
        for i, k in enumerate(ks):
            print(f'k = {k}')

            empirical_error, true_error, _ = self.erm(points, m, k)

            errors[i] = [empirical_error, true_error]

        plt.xlabel('k')
        plt.ylabel('error')

        empirical_errs = errors[:,0]
        true_errs = errors[:,1]
        plt.plot(ks, empirical_errs, 'x', label = 'empirical error')
        plt.plot(ks, true_errs, 'o', label = 'true error')
        plt.legend(loc='upper right')

        plt.savefig('d.png')

        min_err = np.argmin(empirical_errs)
        return list(ks)[min_err]

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        from operator import add
        errors = np.ndarray((math.floor((k_last + 1 - k_first) / step), 2))
        ks = range(k_first, k_last + 1, step)
        points = self.sample_from_D(m)
        points = points[points[:,0].argsort()] # sort by x
        for i, k in enumerate(ks):
            print(f'k = {k}')

            empirical_error, true_error, _ = self.erm(points, m, k)

            errors[i] = [empirical_error, true_error]

        plt.xlabel('k')
        plt.ylabel('error')

        empirical_errs = errors[:,0]
        true_errs = errors[:,1]
        penalties = [self.penalty(m, k) for k in ks]
        penalty_emp_err_sum = list(map(add, empirical_errs, penalties))
        plt.plot(ks, empirical_errs, 'x', label = 'empirical error')
        plt.plot(ks, true_errs, 'o', label = 'true error')
        plt.plot(ks, penalties, 'o', label = 'ERM penalty')
        plt.plot(ks, penalties, 'o', label = 'ERM penalty')
        plt.plot(ks, penalty_emp_err_sum, 'o', label = 'ERM penalty + emipirical error')
        plt.legend(loc='upper right')

        plt.savefig('e.png')

        min_err = np.argmin(penalty_emp_err_sum)
        return list(ks)[min_err]

    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        ks = range(1, 11)
        holdout_errs = [0] * 10
        points = self.sample_from_D(m)
        index = int(0.2 * m)
        for i in range(T):
            np.random.shuffle(points)
            test_points = points[index:]
            test_points = test_points[test_points[:,0].argsort()] # sort by x
            holdout = points[:index]
            for k in ks:
                print(f'k = {k}')

                empirical_error, true_error, best_intervals = self.erm(test_points, m, k)

                holdout_err = self.holdout_error(holdout, best_intervals)
                holdout_errs[k - 1] = holdout_err

        min_err = np.argmin(holdout_errs)
        return list(ks)[min_err]

    #################################
    # Place for additional methods


    #################################

    def intersection(self, first_interval, second_interval):
        (l1_start, l1_end) = first_interval
        (l2_start, l2_end) = second_interval
        if (l2_start < l1_start < l2_end) or (l2_start < l1_end < l2_end) \
                or (l1_start < l2_start < l1_end) or (l1_start < l2_end < l1_end):
                    start = l1_start if l1_start > l2_start else l2_start
                    end = l2_end if l1_end > l2_end else l1_end
                    return start, end
        return

    def intervals_complement(self, line_intervals):
        line_intervals.insert(0, (0, 0))
        line_intervals.append((1, 1))
        complements = []
        for i in range(len(line_intervals) - 1):
            complements.append((line_intervals[i][1], line_intervals[i + 1][0]))

        return complements

    def calculate_true_error(self, best_intervals):
        error = 0

        intervals_complement = self.intervals_complement(best_intervals)

        true_intervals = [(0, 0.2), (0.4, 0.6), (0.8, 1)]
        intersections = [self.intersection(i1, i2) for i1, i2 in itertools.product(best_intervals, true_intervals)]
        intersections = [i for i in intersections if i is not None]
        error += sum(x_end - x_start for x_start, x_end in intersections) * 0.2
        intersections = [self.intersection(i1, i2) for i1, i2 in itertools.product(intervals_complement, true_intervals)]
        intersections = [i for i in intersections if i is not None]
        error += sum(x_end - x_start for x_start, x_end in intersections) * 0.8

        true_intervals = [(0.2, 0.4), (0.6, 0.8)]
        intersections = [self.intersection(i1, i2) for i1, i2 in itertools.product(best_intervals, true_intervals)]
        intersections = [i for i in intersections if i is not None]
        error += sum(x_end - x_start for x_start, x_end in intersections) * 0.9
        intersections = [self.intersection(i1, i2) for i1, i2 in itertools.product(intervals_complement, true_intervals)]
        intersections = [i for i in intersections if i is not None]
        error += sum(x_end - x_start for x_start, x_end in intersections) * 0.1

        return error

    def calculate_empirical_error(self, m, errors):
        return errors / m

    def erm(self, points, m, k):
        xs = points[:,0]
        ys = points[:,1]

        best_intervals, test_errors = intervals.find_best_interval(xs, ys, k)

        empirical_error = self.calculate_empirical_error(m, test_errors)
        true_error = self.calculate_true_error(best_intervals)

        return empirical_error, true_error, best_intervals

    def penalty(self, m, k):
        delta = 0.1
        d = 2 * k

        penalty = np.sqrt(8 / m * (d * np.log(2 * np.e * m / d) + np.log(4 / delta)))

        return penalty

    def holdout_error(self, holdout, best_intervals):
        errs = 0
        for x, y in holdout:
            for start, end in best_intervals:
                if start <= x < end:
                    if y == 0:
                        errs += 1
                    break
            else:
                if y == 1:
                    errs += 1
        return errs


if __name__ == '__main__':
    ass = Assignment2()
    # ass.draw_sample_intervals(100, 3)
    # ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    # ass.experiment_k_range_erm(1500, 1, 20, 1)
    k = ass.experiment_k_range_erm(500, 1, 20, 1)
    print(k)
    k = ass.experiment_k_range_srm(500, 1, 20, 1)
    print(k)
    # k = ass.cross_validation(500, 3)

