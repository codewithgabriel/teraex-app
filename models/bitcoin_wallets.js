import { Schema, Types, model } from "mongoose";

const BtcWalletsSchema = new Schema({
  privateKey: { type: String, required: true }, // Binary private key
  publicKey: { type: String, required: true },  // Binary public key
  address: { type: String, required: true, unique: true }, // Ensure address is unique
  owner: { type: Types.ObjectId, ref: "users", required: true }, // Reference to users
  wif: { type: String, required: true }, // Wallet Import Format private key
  symbol: { type: String, default: "BTC" }, // Optional field with default
  mnemonic: { type: String, required: true }, // Seed phrase
});

// Add an index for faster queries on owner
//BtcWalletsSchema.index({ owner: 1 });

const BitcoinWallets = model("btc_wallets", BtcWalletsSchema);
export default BitcoinWallets;
