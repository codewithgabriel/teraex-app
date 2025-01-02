import { Schema, Types, model } from "mongoose";

const BtcWalletsSchema = new Schema({
  privateKey: { type: String, required: true },
  publicKey: { type: String, required: true },
  address: { type: String, required: true },
  owner: { type: Types.ObjectId, required: true },
  wif: { type: String, required: true },
  symbol: {type: String , default: "BTC"}
});

const BtcWallets = model("btc_wallets", BtcWalletsSchema);
export default BtcWallets;
