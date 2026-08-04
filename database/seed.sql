-- Hotels

INSERT INTO Hotels VALUES
(1,'Aurelia Alexandria','Alexandria','Corniche Road','035555555'),
(2,'Aurelia Cairo','Cairo','Nasr City','022222222'),
(3,'Aurelia Luxor','Luxor','Nile Street','095333333');

-- Rooms

INSERT INTO Rooms VALUES
(101,1,'101','Standard',2,1200,'Available'),
(102,1,'102','Deluxe',2,1800,'Occupied'),
(103,1,'103','Suite',4,3500,'Maintenance'),
(201,2,'201','Standard',2,1300,'Available'),
(202,2,'202','Deluxe',2,1900,'Occupied'),
(203,2,'203','Suite',4,3700,'Available'),
(301,3,'301','Standard',2,1100,'Available'),
(302,3,'302','Deluxe',2,1700,'Occupied'),
(303,3,'303','Suite',4,3400,'Available');

-- Guests

INSERT INTO Guests VALUES
(1,'Ahmed Ali','ahmed@gmail.com','01011111111','Egyptian','Gold'),
(2,'Sara Mohamed','sara@gmail.com','01022222222','Egyptian','VIP'),
(3,'John Smith','john@gmail.com','01033333333','American','Regular'),
(4,'Mariam Hassan','mariam@gmail.com','01044444444','Egyptian','Silver'),
(5,'Omar Adel','omar@gmail.com','01055555555','Egyptian','Regular'),
(6,'Emily Brown','emily@gmail.com','01066666666','British','Gold');

-- Staff

INSERT INTO Staff VALUES
(1,1,'Ali Hassan','Manager','manager.alex@aurelia.com'),
(2,1,'Nour Ahmed','Receptionist','reception.alex@aurelia.com'),
(3,2,'Mohamed Samir','Manager','manager.cairo@aurelia.com'),
(4,2,'Salma Adel','Receptionist','reception.cairo@aurelia.com'),
(5,3,'Khaled Ibrahim','Manager','manager.luxor@aurelia.com'),
(6,3,'Laila Mostafa','Receptionist','reception.luxor@aurelia.com');

-- Reservations

INSERT INTO Reservations VALUES
(1,1,102,'2026-08-01','2026-08-05','Confirmed',7200),
(2,2,103,'2026-08-02','2026-08-06','Confirmed',14000),
(3,3,202,'2026-08-03','2026-08-07','Confirmed',7600),
(4,4,301,'2026-08-05','2026-08-09','Checked In',4400),
(5,5,302,'2026-08-07','2026-08-10','Confirmed',5100),
(6,6,203,'2026-08-08','2026-08-12','Confirmed',14800);

-- Recovery Requests

INSERT INTO Recovery_Requests VALUES
(1,2,'Overbooking','High','Pending',2,'2026-07-30 09:30:00'),
(2,3,'Maintenance','High','In Progress',4,'2026-07-30 10:15:00'),
(3,4,'Room Not Ready','Medium','Resolved',2,'2026-07-30 11:00:00'),
(4,5,'VIP Conflict','High','Pending',6,'2026-07-30 11:45:00'),
(5,6,'Power Outage','Critical','Pending',6,'2026-07-30 12:30:00');

-- Room Transfers

INSERT INTO Room_Transfers VALUES
(1,1,103,203,'Overbooking Recovery','Approved',1),
(2,2,202,201,'Maintenance Recovery','Approved',3),
(3,4,302,303,'VIP Upgrade','Pending',5);

-- Compensations

INSERT INTO Compensations VALUES
(1,1,'Room Upgrade',0,'Approved',1,'2026-07-30 10:00:00'),
(2,2,'Discount',500,'Approved',3,'2026-07-30 10:30:00'),
(3,3,'Meal Voucher',250,'Approved',1,'2026-07-30 11:30:00'),
(4,4,'Free Night',3500,'Pending',5,NULL),
(5,5,'Refund',1000,'Pending',5,NULL);
