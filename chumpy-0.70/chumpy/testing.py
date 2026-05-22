from . import ch
import numpy as np

fn1 = 'assert_allclose', 'assert_almost_equal', 'assert_approx_equal', 'assert_array_almost_equal', 'assert_array_almost_equal_nulp', 'assert_array_equal', 'assert_array_less', 'assert_array_max_ulp', 'assert_equal', 'assert_no_warnings', 'assert_string_equal'
fn2 = 'assert_raises', 'assert_warns'

# These are unhandled
fn3 = 'build_err_msg', 'dec', 'decorate_methods', 'decorators', 'division', 'importall', 'jiffies', 'measure', 'memusage', 'nosetester', 'numpytest', 'print_assert_equal', 'print_function', 'raises', 'rand', 'run_module_suite', 'rundocs', 'runstring', 'test', 'utils', 'verbose'

__all__ = fn1 + fn2

for rtn in fn1:
    exec('def %s(*args, **kwargs) : return np.testing.%s(np.asarray(args[0]), np.asarray(args[1]), *args[2:], **kwargs)' % (rtn, rtn))

for rtn in fn2:
    exec('def %s(*args, **kwargs) : return np.testing.%s(*args, **kwargs)' % (rtn, rtn))



if __name__ == '__main__':
    main()