import { Schema, Types, model } from "mongoose";

const BtcWalletsSchema = new Schema({
  privateKey: { type: String, required: true },
  publicKey: { type: String, required: true },
  address: { type: String, required: true },
  owner: { type: Types.ObjectId, required: true },
  wif: { type: String, required: true },
});

const BtcWallets = model("BTC_wallets", BtcWalletsSchema);

export default BtcWallets;
