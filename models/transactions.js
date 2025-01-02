import { Schema, model , Types } from "mongoose";

const UsersSchema = new Schema({
  from: { type: String, required: true, unique: true },
  to: { type: String, required: true, unique: true },
  amount: { type: String, required: true },
  date: { type: number , default: false },
  status: {type: String }
});


const Users = model("txs", UsersSchema);
export default Users;
