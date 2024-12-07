import { Router } from "express";
var router = Router();
import Users from "../../models/users.js";
import { createBtcWallet } from "./btc/create_btc_wallet.js";
import {
  isText,
  isPassword,
  isEmail,
  getMaxServerTimeout,
  hashPassword,
} from "../../utils/utlities.js";
import BtcWallets from "../../models/btc_wallets.js";
import pgk from "jsonwebtoken";

const { sign } = pgk;

/* create new user account. */
const signupRouter = router.use("/", async function (req, res, next) {
  req.setTimeout(getMaxServerTimeout());
  try {
    const { fullname, password, email } = req.body;
    //step 1. validate user infos for correct format
    if (!(isText(fullname) && isPassword(password) && isEmail(email)))
      throw { reason: "Invalid Token input" };

    //step 2. generate btc wallet
    const { privateKey, publicKey, address, wif, mnemonic } = createBtcWallet();

    const _hashedPassword = await hashPassword(password);

    //step 3. save user
    let user = new Users({ fullname, password: _hashedPassword, email });
    let savedUser = await user.save();
    // console.log(savedUser)

    //step 4. save user btc wallet
    const btcWallet = new BtcWallets({
      privateKey,
      publicKey,
      address,
      wif,
      mnemonic,
      owner: savedUser._id,
    });
    const savedbtcWallet = await btcWallet.save();
    // console.log(savedbtcWallet);

    res.send({
      error: false,
      reason: "User account created",
      type: "USER_ACCT_CREATE",
    });
    res.end();
  } catch (err) {
    if (err.code == 11000) {
      res.send({
        error: true,
        reason: "User already exist",
        type: "USER_ACCT_CREATE_ERR",
      });
    } else {
      res.send({
        error: true,
        reason: err.reason,
        type: "USER_ACCT_CREATE_ERR",
      });
    }
    res.end();
    // log error
    console.log(err);
  }
});

export default signupRouter;
