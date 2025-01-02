import { Schema, Types, model } from "mongoose";

const TeraWalletSchema = new Schema({
  balance: { type: Number, required: true },
  address: { type: String, required: true },
  owner: { type: Types.ObjectId, required: true , unique: true},
});

const TeraWallets = model("tera_wallets", TeraWalletSchema);

export default TeraWallets;
