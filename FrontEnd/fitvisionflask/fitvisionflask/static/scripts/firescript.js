// Import the functions you need from the SDKs you need

// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyAL3E_w1gaR_qaXIw-etd1vVQYbBTctPKs",
    authDomain: "evac-77db5.firebaseapp.com",
    projectId: "evac-77db5",
    storageBucket: "evac-77db5.appspot.com",
    messagingSenderId: "832522615412",
    appId: "1:832522615412:web:57d363bc5a2422fb816f21"
};

// // Initialize Firebase
firebase.initializeApp(firebaseConfig);
// // const auth = getAuth();


// var signInButton = document.getElementById("signInButton");
// var signUpButton = document.getElementById("signUpButton");

// Add event listeners to the sign in and sign up buttons
const Login = () => {
    var email = document.getElementById("userEmail");
    var password = document.getElementById("userPassword");

    signInButton.addEventListener("click", function () {
        // Sign in the user using Firebase's signInWithEmailAndPassword method
        firebase.auth().signInWithEmailAndPassword(email.value, password.value)
            .then(function () {
                console.log("Han valid tak barabar hai ")
                // Redirect the user to the protected resources page
                window.location.href = "index.html";

            })
            .catch(function (error) {
                // Show an error message
                console.log("Gadbad")
                alert(error.message);
            });
    });

}

var signInButton = document.getElementById("signInButton");
if (signInButton) {
    signInButton.addEventListener("click", Login);
}



