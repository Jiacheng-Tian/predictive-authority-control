from __future__ import annotations

import unittest


class OracleContractTest(unittest.TestCase):
    def test_generic_settings_allow_horizon_one_but_formal_contract_rejects_it(self):
        from pac.authority.oracle import OracleSettings, validate_formal_oracle_contract

        validate_formal_oracle_contract(OracleSettings(horizon=20))
        settings = OracleSettings(horizon=1)
        with self.assertRaises(ValueError):
            validate_formal_oracle_contract(settings)
