import { Router } from "express";
let router = Router();
import { createBitcoinWallet } from "../../tokens/bitcoin.js";
import { createEthereumWallet } from "../../tokens/ethereum.js";
import jwtValidator from "../../utils/jwt_validator.js";
import { signUserAuthToken } from "../../utils/utlities.js";
import mongoose from "mongoose";

// import utils
import {
  isText,
  isPassword,
  isEmail,
  getMaxServerTimeout,
  hashPassword,
} from "../../utils/utlities.js";
//import all models
import Users from "../../models/users.js";
import BtcWallets from "../../models/bitcoin_wallets.js";
import NGNWallet from "../../models/ngn_wallets.js";
import TeraWallets from "../../models/tera_wallets.js";
import EthereumWallets from "../../models/ethereum_wallets.js";


import {
  INPUT_VALID_ERR,
  USER_ACCT_ALREADY_EXISTS,
  USER_ACCT_CREATE_ERR,
  USER_ACCT_CREATE_SUCCESS,
  WALLETS_GENERATE_ERROR,
} from "../../utils/states.js";

/* create new user account. */
const signupRouter = router.use("/", async function (req, res, next) {

  req.setTimeout(getMaxServerTimeout());
  // start a new mongoose session
  const session = await mongoose.startSession();
  session.startTransaction();


  try {
    const { fullname, password, email } = req.body;
    // validate user info for correct format
    if (!(isText(fullname) && isPassword(password) && isEmail(email)))
      throw { status: INPUT_VALID_ERR };
    const _hashedPassword = await hashPassword(password);
    // save user
    let user = new Users({ fullname, password: _hashedPassword, email });
    let savedUser = await user.save({ session });

    /* generate defaults  wallets  (BTC, ETH, SOL , NGN (virtual wallet))*/
    await generateDefaultWallets(savedUser, session);
    const { _id } = user;

    const payload = {
      id: _id,
    };
    // sign user's payload
    let signedPayload = await signUserAuthToken(payload);

    // commit the transaction
    await session.commitTransaction();
    session.endSession();

    res.send({
      error: false,
      status: USER_ACCT_CREATE_SUCCESS,
      authToken: signedPayload,
    });
    res.end();

  } catch (err) {
    // rollball all transactions if an error occurs
    await session.abortTransaction();
    session.endSession();

    if (err.code == 11000) {
      res.send({
        error: true,
        status: USER_ACCT_ALREADY_EXISTS,
      });
    } else {
      res.send({
        error: true,
        status: USER_ACCT_CREATE_ERR,
      });
    }
  }



});




    
async function generateDefaultWallets(savedUser, session) {
  try {
        const { privateKey, publicKey, address, wif, mnemonic } = createBitcoinWallet();
        // generate and save btc wallet
        const btcWallet = new BtcWallets({
          privateKey,
          publicKey,
          address,
          wif,
          mnemonic,
          owner: savedUser._id,
        });
        await btcWallet.save({ session });

        // generate and save ethereum wallet
        const {  address: ethAddress ,  privateKey: ethPrivateKey, publicKey: ethPublicKey, mnemonic: ethMnemonic } = createEthereumWallet();
        const ethWallet = new EthereumWallets({ 
          address: ethAddress,
          privateKey: ethPrivateKey,
          publicKey: ethPublicKey,
          address: ethAddress,
          mnemonic: ethMnemonic,
          owner: savedUser._id,
        });
        await ethWallet.save({ session });


        //generate and save virtual ngn wallet
        const ngnWallet = new NGNWallet({
          balance: 0,
          address: `ngn_${savedUser._id}`,
          owner: savedUser._id,
        });
        await ngnWallet.save({ session });

        // generate and save virtual tera wallet
        const teraWallet = new TeraWallets({
          balance: 0,
          address: `tera_${savedUser._id}`,
          owner: savedUser._id,
        });
        await teraWallet.save({ session });
  } catch (err) {
    console.error("Error generating default wallets:", err);
    throw { status: WALLETS_GENERATE_ERROR };

  }
}

export default signupRouter
