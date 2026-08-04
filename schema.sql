CREATE TABLE Hotels (
    hotel_id INT PRIMARY KEY,
    hotel_name VARCHAR(100) NOT NULL,
    city VARCHAR(100),
    address VARCHAR(255),
    phone VARCHAR(20)
);

CREATE TABLE Rooms (
    room_id INT PRIMARY KEY,
    hotel_id INT NOT NULL,
    room_number VARCHAR(10) NOT NULL,
    room_type VARCHAR(50),
    capacity INT,
    price_per_night DECIMAL(10,2),
    room_status VARCHAR(30),
    FOREIGN KEY (hotel_id) REFERENCES Hotels(hotel_id)
);

CREATE TABLE Guests (
    guest_id INT PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    email VARCHAR(100),
    phone VARCHAR(20),
    nationality VARCHAR(50),
    loyalty_level VARCHAR(20)
);

CREATE TABLE Staff (
    staff_id INT PRIMARY KEY,
    hotel_id INT NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    role VARCHAR(50),
    email VARCHAR(100),
    FOREIGN KEY (hotel_id) REFERENCES Hotels(hotel_id)
);

CREATE TABLE Reservations (
    reservation_id INT PRIMARY KEY,
    guest_id INT NOT NULL,
    room_id INT NOT NULL,
    check_in DATE,
    check_out DATE,
    reservation_status VARCHAR(30),
    total_price DECIMAL(10,2),
    FOREIGN KEY (guest_id) REFERENCES Guests(guest_id),
    FOREIGN KEY (room_id) REFERENCES Rooms(room_id)
);

CREATE TABLE Recovery_Requests (
    request_id INT PRIMARY KEY,
    reservation_id INT NOT NULL,
    issue_type VARCHAR(50),
    priority VARCHAR(20),
    request_status VARCHAR(30),
    created_by INT,
    created_at TIMESTAMP,
    FOREIGN KEY (reservation_id) REFERENCES Reservations(reservation_id),
    FOREIGN KEY (created_by) REFERENCES Staff(staff_id)
);

CREATE TABLE Room_Transfers (
    transfer_id INT PRIMARY KEY,
    request_id INT NOT NULL,
    from_room_id INT,
    to_room_id INT,
    transfer_reason VARCHAR(100),
    transfer_status VARCHAR(30),
    approved_by INT,
    FOREIGN KEY (request_id) REFERENCES Recovery_Requests(request_id),
    FOREIGN KEY (from_room_id) REFERENCES Rooms(room_id),
    FOREIGN KEY (to_room_id) REFERENCES Rooms(room_id),
    FOREIGN KEY (approved_by) REFERENCES Staff(staff_id)
);

CREATE TABLE Compensations (
    compensation_id INT PRIMARY KEY,
    request_id INT NOT NULL,
    compensation_type VARCHAR(50),
    amount DECIMAL(10,2),
    approval_status VARCHAR(30),
    approved_by INT,
    approved_at TIMESTAMP,
    FOREIGN KEY (request_id) REFERENCES Recovery_Requests(request_id),
    FOREIGN KEY (approved_by) REFERENCES Staff(staff_id)
);
