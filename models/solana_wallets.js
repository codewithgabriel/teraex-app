import { Schema, Types, model } from "mongoose";

const SolanaWalletsSchema = new Schema({
  privateKey: { type: String, required: true }, // Binary private key
  publicKey: { type: String, required: true },  // Binary public key
  owner: { type: Types.ObjectId, ref: "users", required: true }, // Reference to users
  symbol: { type: String, default: "SOL" }, // Optional field with default
  mnemonic: { type: String, required: true }, // Seed phrase
});

// Add an index for faster queries on owner
//BtcWalletsSchema.index({ owner: 1 });

const SolanaWallets = model("solana_wallets", SolanaWalletsSchema);
export default SolanaWallets;
