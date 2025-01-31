export const ENDPOINT = "/api/v1";

// authentication states
export const USER_ACCT_CREATE_ERR = "Error Creating User Account";
export const USER_ACCT_ALREADY_EXISTS = "User Account Already Exists";
export const USER_ACCT_CREATE_SUCCESS = "User Account Created Successfully"
export const USER_ACCT_AUTH_ERR = "User Authentication Error"
export const USER_ACCT_AUTH_SUCCESS = "User Authentication Success"


// transactions status 
export const TX_PROCESSING  = "Transaction Processing";
export const TX_RECEIVED = "Transaction Received"
export const TX_SEND_SUCCESS = "Transaction Send Sucess"
export const TX_FAILED = "Transaction Failed"
export const TX_CREDITED = "Transaction Credited"
export const TX_DEBITEDD = "Trasaction Debited"
export const TX_CONFIRMED = "Transaction Confirmed"
export const TX_NOT_CONFIRMED = "Transaction Not Confirmed"


// Input Validation status 
export const INPUT_VALID_ERR  = "Invalid Input tokens"
export const PASSWD_MISMATCH_ERR = "Mismatch password"
export const PASSWD_WEEK_ERR = "Weak Password"


// user account status 
export const USER_ACCT_BLOCKED = "User Account Blocked"

// notification states/status 
export const NEW_INCOMING_MSG = "New Incoming Message"
export const MSG_SENT = "Message Sent"


//status Relating to wallet 
export const WALLETS_GENERATE_ERROR = "Error Generating Wallet"
export const WALLETS_GENERATE_SUCESS = "Walets Generated Succesfully"
export const WALLET_INFO_ERROR = "Error Selecting Wallet"
export const WALLET_INFO_SUCCESS = "Wallet Selected Successfully"
export const WALLET_INFO_ERROR_UNKNOWN =  "Unknown Error Selecting Wallet"


//http status 
export const ACCEPT = "accept"
export const REJECT = "reject"
export const IDLE  = "idle"
export const ERROR = 'error'
export const SUCCESS = 'success'
export const PENDING = 'pending'


//http response status code
export const OKAY_CODE = 200;
export const PROCESSING_CODE = 201;
export const NOT_FOUND_CODE = 404;
export const INTER_SERVER_ERR_CODE = 500;
export const FORBIDEN_CODE = 501;
