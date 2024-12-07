import * as bitcoin from "bitcoinjs-lib"; // Import bitcoinjs-lib // Import bip32
import * as bip39 from "bip39"; // Import bip39 for seed generation
import { BIP32Factory } from "bip32";

import * as ecc from "tiny-secp256k1";
import express from "express";
import jwtValidator from "../../../utils/jwt_validator.js";
import BtcWallets from "../../../models/btc_wallets.js";

const route = express.Router();
route.use(jwtValidator);
export default route.post("/", function (req, res, next) {
  try {
    // step 1. generate btc wallet
    const { 
     privateKey,
     publicKey,
     wif,
     address,
     mnemonic } = createBtcWallet();

    //step 2. save walleted
    let wallet = new BtcWallets({
      privateKey,
      publicKey,
      address,
      wif,
      mnemonic,
      owner: req.user._id,
    });
    wallet.save();

    // step 2. send response
    res.send({
      error: false,
      message: "BTC Wallet generated",
      type: "BTC_WALLET_GENERATED",
    });
    res.end();
  } catch (err) {
    res.send({
      error: true,
      message: err.message,
      type: "BTC_CREATE_WALLET_ERR",
    });
    res.end();
  }
});

export function createBtcWallet() {
  // Create the BIP32 instance using the factory and tiny-secp256k1 for elliptic curve operations
  const bip32 = BIP32Factory(ecc);

  // Step 1: Generate a mnemonic phrase (using bip39)
  const mnemonic = bip39.generateMnemonic();
  //console.log("Mnemonic:", mnemonic);  // Important to back this up!

  // Step 2: Generate seed from the mnemonic
  const seed = bip39.mnemonicToSeedSync(mnemonic);

  // Step 3: Use bip32 to derive the HD node from the seed
  const root = bip32.fromSeed(seed, bitcoin.networks.testnet); // Use testnet, or bitcoin.networks.bitcoin for mainnet

  // Step 4: Derive a child key (e.g., first account, first address)
  const account = root.derivePath("m/44'/1'/0'/0/0"); // BIP44 standard path

  // Step 5: Get the private key, public key, WIF, and address
  const privateKey = account.privateKey.toString("hex");
  const publicKey = account.publicKey.toString("hex");
  const wif = account.toWIF(); // Wallet Import Format of the private key

  const { address } = bitcoin.payments.p2pkh({
    pubkey: account.publicKey,
    network: bitcoin.networks.testnet, // Or bitcoin.networks.bitcoin for mainnet
  });

  return {
    privateKey,
    publicKey,
    wif,
    address,
    mnemonic,
  };
}
