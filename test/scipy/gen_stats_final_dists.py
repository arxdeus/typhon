# Copyright (C) 2026 the Typhon authors
# Generates test/scipy/test_stats_final_dists*.py with scipy 1.18.1:
#   python gen_stats_final_dists.py <group>   (a: kstwo/geninvgauss/jf_skew_t, b: landau/vonmises_line/dpareto_lognorm/genhyperbolic)
import sys, os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gen_dists_expected.py')).read()
head = src.split("if sys.argv[1] in DGROUPS:")[0]
sys.argv = [sys.argv[0], sys.argv[1]]
ns = {}
exec(head, ns)
ns['GROUPS']['fa'] = [('kstwo', (10,), 1e-9), ('geninvgauss', (0.8, 1.7), 1e-8),
                      ('jf_skew_t', (3.5, 4.5), 1e-10)]
ns['GROUPS']['fb'] = [('landau', (), 1e-9), ('vonmises_line', (1.8,), 1e-10),
                      ('dpareto_lognorm', (0.3, 0.6, 2.5, 3.5), 1e-9),
                      ('genhyperbolic', (0.7, 1.9, 0.6), 1e-7)]
ns['emit'](sys.argv[1])
