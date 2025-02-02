import { Router } from "express";
var router = Router();

import Tokens from "../../models/tokens.js";
import jwtValidator from "../../utils/jwt_validator.js";

/* GET users listing. */
router.use(jwtValidator);
router.get("/", function (req, res, next) {
  BitcoinWallets.find({}, function (err, wallets) {
    if (err) {
      return next(err);
    }
    res.json(wallets);
  });
});

export default router;
