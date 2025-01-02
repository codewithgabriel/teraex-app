import { Schema, Types, model } from "mongoose";

const WithdrawalInfoSchema = new Schema({
  balance: { type: String, required: true },
  address: { type: String, required: true },
  owner: { type: Types.ObjectId, required: true },
});

const WithdrawalInfo = model("withdrawal_accounts", WithdrawalInfoSchema);

export default WithdrawalInfo;
