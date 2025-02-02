import { Schema, Types, model } from "mongoose";

const EthereumWalletsSchema = new Schema({
  privateKey: { type: String, required: true }, // Binary private key
  publicKey: { type: String, required: true },  // Binary public key
  owner: { type: Types.ObjectId, ref: "users", required: true }, // Reference to users
  symbol: { type: String, default: "ETH" }, // Optional field with default
  mnemonic: { type: String, required: true }, // Seed phrase
});

// Add an index for faster queries on owner
//BtcWalletsSchema.index({ owner: 1 });

const EthereumWallets = model("ethereum_wallets", EthereumWalletsSchema);
export default EthereumWallets;
