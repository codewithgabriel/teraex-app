import { Schema, model , Types } from "mongoose";

const UsersSchema = new Schema({
  from: { type: String, required: true, unique: true },
  to: { type: String, required: true, unique: true },
  amount: { type: String, required: true },
  date: { type: number , default: false },
  type: { type: String},
  status: {type: String }
});


const Transactions = model("transactions", UsersSchema);
export default Transactions;
