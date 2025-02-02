import { Schema, Types, model } from "mongoose";

const NGNWalletSchema = new Schema({
  balance: { type: Number, required: true ,  default: 0},
  address: { type: String, required: true },
  owner: { type: Types.ObjectId, required: true , unique: true },
});

const NGNWallet = model("ngn_wallets", NGNWalletSchema);

export default NGNWallet;
