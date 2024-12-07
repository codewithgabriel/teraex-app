import { configDotenv } from "dotenv";
configDotenv();
import pkg from "jsonwebtoken";
import bcrypt from "bcrypt";
const { sign } = pkg;

let { APP_SECRET_KEY, BTC_RATE, BTC_UNIT, MAX_TIMEOUT } = process.env;

export function isEmail(email) {
  var reg = /^[A-Z0-9._%+-]+@([A-Z0-9-]+\.)+[A-Z]{2,4}$/i;
  if (reg.test(email)) {
    return true;
  } else {
    return false;
  }
}

export function isText(text) {
  var reg = /^[A-Za-z]/i;
  if (reg.test(text)) {
    return true;
  } else {
    return false;
  }
}

export function isPassword(pass) {
  try {
    if (
      pass.length >= 8 &&
      /((?=.*\d)(?=.*[a-z])(?=.*[A-Z])(?=.*[\W]).{6,20})/.test(pass)
    ) {
      return true;
    } else {
      return false;
    }
  } catch (e) {
    console.log(e.message);
  }
}
export function getMaxServerTimeout() {
  const max_timeout = parseInt(MAX_TIMEOUT);
  return max_timeout;
}

function getBtcUnit() {
  const btc_unit = parseFloat(BTC_UNIT);
  return btc_unit;
}

function getBtcRate() {
  const btc_rate = parseFloat(BTC_RATE);
  return btc_rate;
}

function getNairaRate(amount) {
  const naira_rate = amount * parseFloat(BTC_RATE);
  return naira_rate;
}

export async function signUserAuthToken(payload) {
  const signedPayload = sign(payload, APP_SECRET_KEY);
  return signedPayload;
}

const saltRounds = 10; // Number of salt rounds
export async function hashPassword(plainPassword) {
  const salt = await bcrypt.genSalt(saltRounds);
  const hash = await bcrypt.hash(plainPassword, salt);
  return hash;
}

export async function verifyPassword(plainPassword, hash) {
  const match = await bcrypt.compare(plainPassword, hash);
  return match;
}
